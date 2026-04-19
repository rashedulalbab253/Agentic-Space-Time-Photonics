import torch
from torch.autograd import grad
import torch.optim as optim
import hash_model as hash_model


class ModelWrapper:
    def __init__(self, args):
        self.args = args

    def sample_points(self, nx, ny, level, scale=1):
        raise NotImplementedError

    def update_points(self):
        pass

    def get_feature_loss(self, min_gap, min_post, sdf_max, samples_per_point):
        raise NotImplementedError

    def per_loop_call(self, i):
        pass

    def after_gradient_call(self, i):
        pass


class ParallelHashModelWrapper(ModelWrapper):
    def __init__(self, args, num_models):
        super().__init__(args)
        self.num_models = num_models
        self.batch_size = args.model["batch_size"]
        self.sdf_model = hash_model.Network(args, num_models).to(args.device)
        
        # Initialize points for all models at once
        self.points = torch.rand([num_models, args.model["batch_size"], 1], 
                               device=args.device, 
                               dtype=torch.float32)
        self.points.requires_grad = True
        
        # Single optimizer for all models
        self.optimizer = torch.optim.Adam(self.sdf_model.parameters(), lr=0.001)
        self.point_optimizer = optim.Adam([self.points], lr=0.01)
        
        # Initialize xy grid for all models
        self.initialize_xy(args.misc["resolution_image"])

    def initialize_xy(self, resolution):
        half_dx = 0.5 / resolution
        xs = torch.linspace(half_dx, 1 - half_dx, resolution, 
                           device=self.args.device)
        self.xy = xs.unsqueeze(1).unsqueeze(0).expand(self.num_models, -1, -1)

    def sample_points(self, nx, level=None, scale=1):
        res_x = int(nx * scale)
        step_x = 1.0 / res_x
        rescale_factor = 4

        x_high_res = torch.linspace(
            0.5 - (step_x * res_x) / 2 + step_x / rescale_factor / 2,
            0.5 + (step_x * res_x) / 2 - step_x / rescale_factor / 2,
            rescale_factor * res_x,
            device=self.args.device
        )
        
        # Add small random perturbation for all models at once
        x_high_res = x_high_res.unsqueeze(0).expand(self.num_models, -1)
        x_high_res = x_high_res + torch.randn_like(x_high_res) * step_x / 16
        
        xy_high = x_high_res.unsqueeze(-1)
        sdf_high = self.sdf_model(xy_high).reshape(self.num_models, rescale_factor * res_x)

        if level is None:
            boundary_high = torch.where(sdf_high > 0, 1, 0).float()
        else:
            boundary_high_1 = torch.where((sdf_high > -level) & (sdf_high < level), 
                                        (sdf_high + level) / (2 * level), 0)
            boundary_high_2 = torch.where(sdf_high > level, 1, 0).float()
            boundary_high = boundary_high_1 + boundary_high_2

        sdf = torch.nn.functional.avg_pool1d(
            sdf_high.unsqueeze(1), 
            rescale_factor, 
            stride=rescale_factor
        ).squeeze(1)
        
        boundary = torch.nn.functional.avg_pool1d(
            boundary_high.unsqueeze(1), 
            rescale_factor, 
            stride=rescale_factor
        ).squeeze(1)

        return sdf, boundary

    def find_zero_point(self):
        for j in range(self.args.model["inner_steps"]):
            self.point_optimizer.zero_grad()
            output = self.sdf_model(self.points)
            loss_points = output.square().sum()
            loss_points.backward()
            self.point_optimizer.step()

        with torch.no_grad():
            # Calculate residuals for all points
            residue = output.square().sum(dim=2)  # [num_models, batch_size]
            
            # Find threshold for each model but use the mean threshold across models
            thresholds = torch.quantile(residue, 0.25, dim=1) * 2  # [num_models]
            mean_threshold = thresholds.mean()
            
            # Apply same threshold to all models
            mask = residue < mean_threshold  # [num_models, batch_size]
            
            # Keep fixed number of points for all models
            points_to_keep = min(self.batch_size // 2, mask.sum(dim=1).min().item())
            
            # Sort points by residue and keep top k
            _, indices = torch.topk(residue, points_to_keep, dim=1, largest=False)
            
            # Gather selected points for all models
            batch_indices = torch.arange(self.num_models, device=self.args.device)
            selected_points = self.points[batch_indices.unsqueeze(1), indices]
            
            # Assume boundary points at 0 and 1
            boundary_points = torch.tensor([[0.0], [1.0]], device=self.args.device)
            boundary_points = boundary_points.unsqueeze(0).expand(self.num_models, -1, -1)
            
            # Combine selected and random points
            self.points = torch.cat([selected_points, boundary_points], dim=1)
            self.points.requires_grad = True
            
        self.point_optimizer = optim.Adam([self.points], lr=1e-3)

    def get_feature_loss(self, min_gap=0.01, min_post=0.01, sdf_max=0.01, samples_per_point=6):
        # Calculate gradients for all points at once
        outputs = self.sdf_model(self.points)  # [num_models, batch_size, 1]
        grad_zero = grad(outputs=outputs.sum(), inputs=self.points, create_graph=True)[0]
        
        # Normalize gradients
        grad_norm = torch.norm(grad_zero, dim=2, keepdim=True)
        normalized_grad = grad_zero / (grad_norm + 1e-8)  # [num_models, batch_size, 1]
        
        # Generate random lengths efficiently
        random_length = torch.rand(
            [self.num_models, samples_per_point * self.points.shape[1], 1],
            device=self.args.device
        )
        
        # Expand points and gradients to match random lengths
        points_expanded = self.points.repeat_interleave(samples_per_point, dim=1)
        grad_expanded = normalized_grad.repeat_interleave(samples_per_point, dim=1)
        
        # Generate positive and negative points in one operation
        positive_points = (points_expanded + grad_expanded * random_length * min_post).detach()
        negative_points = (points_expanded - grad_expanded * random_length * min_gap).detach()
        
        # Evaluate all points in parallel
        output_positive = self.sdf_model(positive_points)
        output_negative = self.sdf_model(negative_points)
        
        # Calculate losses efficiently
        loss_positive = (-output_positive + sdf_max * random_length).pow(2).mean()
        loss_negative = (output_negative + sdf_max * random_length).pow(2).mean()
        
        return (loss_positive + loss_negative) * self.num_models

    def update_points(self):
        with torch.no_grad():
            # Evaluate all grid points
            output = self.sdf_model(self.xy)  # [num_models, resolution, 1]
            residue = output.square().sum(dim=2)  # [num_models, resolution]
            
            # Use same threshold for all models
            threshold = 1e-5
            mask = residue < threshold  # [num_models, resolution]
            
            # Fixed number of points to select from grid
            points_from_grid = self.batch_size // 4
            
            # Random selection of grid points
            grid_points = torch.zeros(
                self.num_models, points_from_grid, 1,
                device=self.args.device
            )
            
            # Select random valid points for each model
            for i in range(self.num_models):
                valid_points = self.xy[i][mask[i]]
                if len(valid_points) > points_from_grid:
                    perm = torch.randperm(len(valid_points), device=self.args.device)
                    grid_points[i] = valid_points[perm[:points_from_grid]]
                else:
                    # If not enough valid points, fill with random points
                    num_valid = len(valid_points)
                    grid_points[i, :num_valid] = valid_points
                    grid_points[i, num_valid:] = torch.rand(
                        points_from_grid - num_valid, 1,
                        device=self.args.device
                    )
            
            # Keep some existing points
            points_to_keep = self.batch_size // 2
            kept_points = self.points[:, :points_to_keep]
            
            # Generate remaining random points
            random_points = torch.rand(
                self.num_models,
                self.batch_size - points_to_keep - points_from_grid,
                1,
                device=self.args.device
            )

            # Combine all points
            self.points = torch.cat([kept_points, grid_points, random_points], dim=1)
            self.points.requires_grad = True
            
        self.point_optimizer = optim.Adam([self.points], lr=0.01)

    def per_loop_call(self, i):
        if i > self.args.threshold:
            for j in range(self.args.model["outer_steps"]):
                self.update_points()
                self.find_zero_point()
                loss = self.get_feature_loss(
                    min_gap=self.args.model["min_gap"],
                    min_post=self.args.model["min_post"],
                    sdf_max=self.args.model["sdf_max"],
                    samples_per_point=self.args.model["samples_per_point"],
                )
                loss = loss * self.args.feature_weight
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()