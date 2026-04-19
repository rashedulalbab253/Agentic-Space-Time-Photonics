import sqlite3
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
import uuid


class DesignDatabase:
    def __init__(self, db_path: str = "designs.db"):
        """Initialize the design database with the specified path."""
        self.db_path = db_path
        self._create_tables_if_not_exist()
        
    def _create_tables_if_not_exist(self):
        """Create the database tables if they don't already exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create designs table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS designs (
                id TEXT PRIMARY KEY,
                design_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                parameters TEXT NOT NULL,  -- JSON string of parameters
                gds_file_path TEXT,
                success INTEGER NOT NULL,
                description TEXT,
                user_id TEXT
            )
            ''')
            
            # Create files table for associated files (plots, etc.)
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS design_files (
                id TEXT PRIMARY KEY,
                design_id TEXT NOT NULL,
                file_type TEXT NOT NULL,
                file_path TEXT NOT NULL,
                description TEXT,
                FOREIGN KEY (design_id) REFERENCES designs (id)
            )
            ''')
            
            conn.commit()
    
    def save_design(self, 
                   design_type: str, 
                   parameters: Dict[str, Any], 
                   gds_file_path: Optional[str] = None,
                   success: bool = True,
                   description: str = "",
                   user_id: Optional[str] = None,
                   associated_files: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Save a design to the database.
        
        Args:
            design_type: Type of design (e.g., "metalens", "deflector")
            parameters: Dictionary of design parameters
            gds_file_path: Path to the GDS file (if generated)
            success: Whether the design completed successfully
            description: Optional description of the design
            user_id: Optional ID of the user who created the design
            associated_files: List of dictionaries with file info: 
                             {"file_type": "...", "file_path": "...", "description": "..."}
                             
        Returns:
            The ID of the saved design
        """
        design_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Insert the design
            cursor.execute(
                "INSERT INTO designs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    design_id,
                    design_type,
                    timestamp,
                    json.dumps(parameters),
                    gds_file_path,
                    1 if success else 0,
                    description,
                    user_id
                )
            )
            
            # Insert associated files if any
            if associated_files:
                for file_info in associated_files:
                    file_id = str(uuid.uuid4())
                    cursor.execute(
                        "INSERT INTO design_files VALUES (?, ?, ?, ?, ?)",
                        (
                            file_id,
                            design_id,
                            file_info.get("file_type", "unknown"),
                            file_info.get("file_path", ""),
                            file_info.get("description", "")
                        )
                    )
                    
            conn.commit()
            
        return design_id
    
    def get_design(self, design_id: str) -> Dict[str, Any]:
        """Retrieve a design by its ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get the design
            cursor.execute("SELECT * FROM designs WHERE id = ?", (design_id,))
            design_row = cursor.fetchone()
            
            if not design_row:
                return {}
            
            design = dict(design_row)
            design["parameters"] = json.loads(design["parameters"])
            design["success"] = bool(design["success"])
            
            # Get associated files
            cursor.execute("SELECT * FROM design_files WHERE design_id = ?", (design_id,))
            design["files"] = [dict(row) for row in cursor.fetchall()]
            
            return design
    
    def search_designs(self, 
                      design_type: Optional[str] = None,
                      success: Optional[bool] = None,
                      user_id: Optional[str] = None,
                      keyword: Optional[str] = None,
                      limit: int = 100) -> List[Dict[str, Any]]:
        """
        Search for designs based on criteria.
        
        Args:
            design_type: Type of design to filter by
            success: Filter by success status
            user_id: Filter by user ID
            keyword: Search in description
            limit: Maximum number of results to return
            
        Returns:
            List of design dictionaries
        """
        query = "SELECT id, design_type, timestamp, gds_file_path, success, description FROM designs WHERE 1=1"
        params = []
        
        if design_type:
            query += " AND design_type = ?"
            params.append(design_type)
            
        if success is not None:
            query += " AND success = ?"
            params.append(1 if success else 0)
            
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
            
        if keyword:
            query += " AND (description LIKE ? OR parameters LIKE ?)"
            params.extend([f"%{keyword}%", f"%{keyword}%"])
            
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, params)
            
            results = []
            for row in cursor.fetchall():
                design = dict(row)
                design["success"] = bool(design["success"])
                results.append(design)
                
            return results
    
    def delete_design(self, design_id: str, delete_files: bool = False) -> bool:
        """
        Delete a design from the database.
        
        Args:
            design_id: ID of the design to delete
            delete_files: Whether to also delete the physical files
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get file paths if deleting files
                if delete_files:
                    cursor.execute("SELECT gds_file_path FROM designs WHERE id = ?", (design_id,))
                    gds_path = cursor.fetchone()
                    
                    cursor.execute("SELECT file_path FROM design_files WHERE design_id = ?", (design_id,))
                    file_paths = [row[0] for row in cursor.fetchall()]
                    
                # Delete from design_files first (foreign key constraint)
                cursor.execute("DELETE FROM design_files WHERE design_id = ?", (design_id,))
                
                # Delete the design
                cursor.execute("DELETE FROM designs WHERE id = ?", (design_id,))
                
                # Delete physical files if requested
                if delete_files:
                    if gds_path and gds_path[0]:
                        try:
                            os.remove(gds_path[0])
                        except OSError:
                            pass
                    
                    for path in file_paths:
                        try:
                            os.remove(path)
                        except OSError:
                            pass
                
                conn.commit()
                return True
        except Exception as e:
            print(f"Error deleting design: {str(e)}")
            return False