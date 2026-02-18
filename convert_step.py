import gmsh
import os
import sys

def convert_step_to_stl(step_file, stl_file):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    
    # Import STEP file
    print(f"Importing {step_file}...")
    try:
        gmsh.merge(step_file)
    except Exception as e:
        print(f"Error importing file: {e}")
        gmsh.finalize()
        return False

    # Synchronize to create the model
    # gmsh.model.occ.synchronize() # Only needed if using OCC kernel explicitly, but merge usually handles it.
    
    # Set mesh options for coarser mesh to reduce complexity
    # 2D mesh algorithm (1=MeshAdapt, 2=Automatic, 5=Delaunay, 6=Frontal-Delaunay)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    # Target mesh size factor (larger value = coarser mesh)
    # Default is usually 1.0. Increasing this will reduce the number of triangles.
    gmsh.option.setNumber("Mesh.MeshSizeFactor", 5.0) 
    # Minimum mesh size
    gmsh.option.setNumber("Mesh.MeshSizeMin", 10.0) 
    # Maximum mesh size
    gmsh.option.setNumber("Mesh.MeshSizeMax", 500.0) 

    # Mesh generation might be needed for STL, but let's try direct export first.
    # Usually for STL we need a surface mesh.
    print("Generating mesh with reduced complexity...")
    try:
        gmsh.model.mesh.generate(2) # Generate 2D mesh (surface mesh)
    except Exception as e:
        print(f"Error generating mesh: {e}")
        gmsh.finalize()
        return False
    
    # Export to STL
    print(f"Exporting to {stl_file}...")
    try:
        gmsh.write(stl_file)
    except Exception as e:
        print(f"Error writing STL: {e}")
        gmsh.finalize()
        return False
    
    gmsh.finalize()
    return True

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 convert.py <input.step> <output.stl>")
        sys.exit(1)
        
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    if not os.path.exists(input_path):
        print(f"Input file not found: {input_path}")
        sys.exit(1)
        
    success = convert_step_to_stl(input_path, output_path)
    if success:
        print("Conversion successful!")
    else:
        print("Conversion failed!")
        sys.exit(1)
        
