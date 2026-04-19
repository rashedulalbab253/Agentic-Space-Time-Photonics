import asyncio
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
import sys
import os
import dotenv

# Add project root to Python path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from tools.material_db.query import MaterialQueryInterface
from core.models.openai import OpenAIModel
# from core.models.anthropic import AnthropicModel
from core.models.base import BaseModel

# Load environment variables
dotenv.load_dotenv()
DEFAULT_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-5.2-2025-12-11")
AVAILABLE_MODELS = ["gpt-5.2-2025-12-11", "gpt-5", "gpt-4o", "gpt-4o-mini"]

if DEFAULT_MODEL_NAME not in AVAILABLE_MODELS:
    AVAILABLE_MODELS.append(DEFAULT_MODEL_NAME)

if DEFAULT_MODEL_NAME not in AVAILABLE_MODELS:
    AVAILABLE_MODELS.append(DEFAULT_MODEL_NAME)

class MaterialDatabaseCLI:
    def __init__(self, db_path: str, model: BaseModel, debug: bool = False, log_dir: str = "logs/materials_chat"):
        self.model = model
        self.log_dir = Path(log_dir) / (model.model_name if model else "default")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize query interface
        self.query_interface = MaterialQueryInterface(
            db_path=db_path,
            model=self.model,
            debug=debug,
            log_dir=self.log_dir
        )
        
        self.conversation_history = []
        self.current_messages = None
        self.current_conversation_id = None

    async def interactive_session(self):
        """Run an interactive query session."""
        print("\nWelcome to the Materials Database Query Interface!")
        print("------------------------------------------------")
        print("Enter your questions about materials and their properties.")
        print("Type 'recommend' to start a material recommendation query.")
        print("Type 'clear' to start a new conversation.")
        print("Type 'quit' to exit.")
        print("------------------------------------------------\n")
        
        while True:
            try:
                command = input("\nYou: ").strip()
                
                if command.lower() == 'quit':
                    break
                    
                if command.lower() == 'clear':
                    self.conversation_history = []
                    self.current_messages = None
                    print("\nConversation history cleared.")
                    continue
                
                # if command.lower() == 'recommend':
                #     # Get application description
                #     description = input("\nDescribe your application: ").strip()
                    
                #     # Get constraints
                #     print("\nEnter constraints (press Enter without input when done)")
                #     print("Format: key=value (e.g., wavelength=1550e-9)")
                    
                #     constraints = {}
                #     while True:
                #         constraint = input("Constraint (or Enter to continue): ").strip()
                #         if not constraint:
                #             break
                            
                #         try:
                #             key, value = constraint.split('=')
                #             # Try to convert value to float or int if possible
                #             try:
                #                 value = float(value)
                #             except ValueError:
                #                 try:
                #                     value = int(value)
                #                 except ValueError:
                #                     pass
                #             constraints[key.strip()] = value
                #         except ValueError:
                #             print("Invalid constraint format. Use key=value")
                #             continue
                    
                #     print("\nQuerying database...")
                #     results = await self.query_interface.recommend_material(
                #         description,
                #         constraints=constraints,
                #         conversation_history=self.conversation_history
                #     )
                else:
                    # Regular query/conversation
                    print("\nProcessing...")
                    results = await self.query_interface.query(
                        command,
                        conversation_history=self.conversation_history
                    )
                    
                # Handle response
                if results.get("status") == "response":
                    message = results.get("message", "No response provided")
                    self.conversation_history.append({
                        "query_explanation": command,
                        "results": message
                    })
                    self.current_messages = results.get("messages")
                    
                    # Print agent's response
                    print(f"\nAgent: {message}")
                
            except KeyboardInterrupt:
                print("\nInteraction interrupted.")
                continue
                
            except Exception as e:
                print(f"\nError: {str(e)}")
                continue
                
        print("\nThank you for using the Materials Database Query Interface!")

    async def query_with_id(self, question: str, conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """Query with optional conversation ID tracking."""
        results = await self.query_interface.query(
            question,
            conversation_id=conversation_id,
            conversation_history=self.conversation_history
        )
        return results

async def main():
    parser = argparse.ArgumentParser(description='Query the materials database')
    parser.add_argument('--db-path', 
                       type=str,
                       default="materials.db",
                       help='Path to the SQLite database file')
    parser.add_argument('--model',
                       type=str,
                       default=DEFAULT_MODEL_NAME,
                       choices=AVAILABLE_MODELS,
                       help='OpenAI model to use')
    parser.add_argument('--debug',
                       action='store_true',
                       default=True,
                       help='Enable debug output')
    
    args = parser.parse_args()
    
    # Check if database exists
    if not Path(args.db_path).exists():
        print(f"Error: Database file not found at {args.db_path}")
        sys.exit(1)
    
    try:
        cli = MaterialDatabaseCLI(
            db_path=args.db_path,
            model=OpenAIModel(
                model_name=args.model,
                api_key=os.getenv("OPENAI_API_KEY")
            ),
            # model=AnthropicModel(
            #     model_name="claude-3-5-sonnet-20240620",
            #     api_key=os.getenv("ANTHROPIC_API_KEY")
            # ),
            debug=args.debug
        )
        await cli.interactive_session()
        
    except Exception as e:
        print(f"Error initializing CLI: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
