import bpy
import numpy as np
import scipy
import os
import mathutils
from mathutils import Vector, Matrix, Quaternion, Euler
import math
from pathlib import Path
from bpy.types import Operator, Panel, Menu
import time

class WORKPIECE_OT_ImportSTL(Operator):
    bl_idname = "workpiece.import_stl"
    bl_label = "Import STL"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        # Set the Unit settings to Metric and mm, scale to 0.001
        bpy.context.scene.unit_settings.system = 'METRIC'
        bpy.context.scene.unit_settings.scale_length = 0.001  # Set scale to mm
        bpy.context.scene.unit_settings.length_unit = 'MILLIMETERS'
        # Set the viewport properties
        bpy.context.space_data.clip_start = 0.1 # Minimum distance to the camera = 0.1 mm
        bpy.context.space_data.clip_end = 10000.0 # Maximum distance to the camera = 10 m
        bpy.context.space_data.overlay.grid_scale = 0.001 # Set grid scale to 1 mm
        bpy.context.space_data.overlay.show_axis_z = True # Show Z axis in the viewport

        stl_file = bpy.context.scene.stl_file
        if not stl_file or not (stl_file.endswith('.stl') or stl_file.endswith('.STL')):
            self.report({'ERROR'}, "Please select a valid STL file")
            return {'CANCELLED'}
        
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()

        t = time.time() # start timer

        try:
            bpy.ops.wm.stl_import(filepath=stl_file)
        except Exception as e:
            self.report({'ERROR'}, f"Failed to import STL: {e}")
            return {'CANCELLED'}
        
        obj = bpy.context.selected_objects[0]
        obj.name = os.path.basename(stl_file)[:-4]
        mesh = obj.data

        self.report({'INFO'}, f"Imported {obj.name} in {time.time() - t:.3f} seconds")
        return {'FINISHED'}
    
class WORKPIECE_OT_Alignment(Operator):
    bl_idname = "workpiece.alignment"
    bl_label = "Align Workpiece"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = "Align the workpiece using canonical alignment methods"
    
    def delete_unused_materials(self):
        """Remove all materials with zero users from the Blender file."""
        for material in bpy.data.materials:
            if material.users == 0:
                self.report({'INFO'}, f"Removing unused material: {material.name}")
                bpy.data.materials.remove(material)

    def execute(self, context):
        obj = bpy.context.active_object
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "No active mesh object found")
            return {'CANCELLED'}

        t = time.time()  # Start timer
        mesh = obj.data
        bpy.ops.object.mode_set(mode='OBJECT')  # Ensure object mode
        bpy.ops.object.select_all(action='DESELECT')  # Deselect all objects
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj  # Set active object
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

        # Delete unused materials before creating new ones
        self.delete_unused_materials()

        if context.scene.alignment_mode == 'AUTO':
            # Step 1: Compute PCA using world space vertices
            vertices = np.array([obj.matrix_world @ v.co for v in mesh.vertices])
            if len(vertices) < 3:
                self.report({'ERROR'}, "Not enough vertices for PCA alignment")
                return {'CANCELLED'}

            # Perform PCA
            mean = np.mean(vertices, axis=0)  # Calculate mean
            centered = vertices - mean  # Center the mesh
            cov_matrix = np.cov(centered.T)  # Calculate covariance matrix
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)  # Eigenvalues and eigenvectors
            order = np.argsort(eigenvalues)[::-1]  # Sort in descending order
            eigenvectors = eigenvectors[:, order]  # Sort eigenvectors (PC1, PC2, PC3)
            
            # Debug: Print eigenvalues and eigenvectors
            self.report({'INFO'}, f"Eigenvalues: {eigenvalues[order]}")
            self.report({'INFO'}, f"Eigenvectors:\n{eigenvectors}")
            self.report({'INFO'}, f"Centroid: {mean}")

            # Check for degenerate PCA
            if np.linalg.matrix_rank(cov_matrix) < 3:
                self.report({'ERROR'}, "Mesh is degenerate (e.g., flat). PCA alignment not possible.")
                return {'CANCELLED'}
            
            # Step 2: Count faces aligned with PC3 and -PC3 (95% accuracy, ~5.74 degrees)
            pc3 = eigenvectors[:, 2]  # Third principal component
            cos_95_percent = np.cos(np.deg2rad(5.74))  # Cosine of 5.74 degrees
            face_normals = np.array([obj.matrix_world @ f.normal for f in mesh.polygons])
            norm_pc3 = pc3 / np.linalg.norm(pc3)
            dots_pc3 = np.dot(face_normals, norm_pc3)
            dots_neg_pc3 = np.dot(face_normals, -norm_pc3)
            # Identify faces aligned with +PC3 and -PC3
            pc3_pos_faces = np.where(dots_pc3 >= cos_95_percent)[0]
            pc3_neg_faces = np.where(dots_neg_pc3 >= cos_95_percent)[0]
            count_pc3 = len(pc3_pos_faces)
            count_neg_pc3 = len(pc3_neg_faces)
            if count_pc3 > count_neg_pc3:
                norm_pc3 = -norm_pc3
                dots_pc3 = np.dot(face_normals, norm_pc3)
                dots_neg_pc3 = np.dot(face_normals, -norm_pc3)
                # Identify faces aligned with +PC3 and -PC3
                pc3_pos_faces = np.where(dots_pc3 >= cos_95_percent)[0]
                pc3_neg_faces = np.where(dots_neg_pc3 >= cos_95_percent)[0]

            # Step 3: Count faces aligned with PC2 and -PC2 (70% accuracy, ~45.58 degrees)
            pc2 = eigenvectors[:, 1]  # Second principal component
            cos_70_percent = np.cos(np.deg2rad(45.58))  # Cosine of 45.58 degrees
            norm_pc2 = pc2 / np.linalg.norm(pc2)
            dots_pc2 = np.dot(face_normals, norm_pc2)
            dots_neg_pc2 = np.dot(face_normals, -norm_pc2)
            # Identify faces aligned with +PC2 and -PC2
            pc2_pos_faces = np.where(dots_pc2 >= cos_70_percent)[0]
            pc2_neg_faces = np.where(dots_neg_pc2 >= cos_70_percent)[0]
            count_pc2 = len(pc2_pos_faces)
            count_neg_pc2 = len(pc2_neg_faces)
            if count_pc2 < count_neg_pc2:
                norm_pc2 = -norm_pc2
                dots_pc2 = np.dot(face_normals, norm_pc2)
                dots_neg_pc2 = np.dot(face_normals, -norm_pc2)
                # Identify faces aligned with +PC2 and -PC2
                pc2_pos_faces = np.where(dots_pc2 >= cos_70_percent)[0]
                pc2_neg_faces = np.where(dots_neg_pc2 >= cos_70_percent)[0]

            # Step 4: Calculate the right-handed PC1, according to the norm_pc3 and norm_pc2
            z_axis = norm_pc3
            y_axis = norm_pc2
            x_axis = np.cross(y_axis, z_axis)  # Compute PC1 (orthogonal to PC2 and PC3)
            x_axis = x_axis / np.linalg.norm(x_axis)  # Normalize
            y_axis = np.cross(z_axis, x_axis)  # Recompute y_axis for right-handed system
            y_axis = y_axis / np.linalg.norm(y_axis)

            # Step 5: Identify faces aligned with +PC1 and -PC1
            cos_80_percent = np.cos(np.deg2rad(45.58))  # Cosine of 45.58 degrees
            dots_pc1 = np.dot(face_normals, x_axis)
            dots_neg_pc1 = np.dot(face_normals, -x_axis)
            pc1_pos_faces = np.where(dots_pc1 >= cos_80_percent)[0]
            pc1_neg_faces = np.where(dots_neg_pc1 >= cos_80_percent)[0]

            # Visualize the principal components
            bpy.ops.object.empty_add(type='ARROWS', radius=10, align='WORLD', location=mean, scale=(1, 1, 1))
            arrows = bpy.context.active_object
            arrows.name = "Principal Components"
            arrows.show_name = True
            rot_matrix = Matrix([x_axis, y_axis, z_axis]).transposed().to_4x4()

            # Apply rotation to the arrows
            arrows.matrix_world = rot_matrix @ Matrix.Translation((0, 0, 0))
            
            # Debug: Print face counts
            self.report({'INFO'}, f"PC3 (+): {count_pc3} faces, PC3 (-): {count_neg_pc3} faces")
            if len(face_normals) > 0:
                self.report({'INFO'}, f"Sample face normals (first 3): {face_normals[:3]}")

            # Step 6: Create vertex groups for +PC3 and -PC3, +PC2 and -PC2, +PC1 and -PC1
            # Remove existing vertex groups if they exist
            for vg_name in ["+PC3", "-PC3", "+PC2", "-PC2", "+PC1", "-PC1"]:
                if vg_name in obj.vertex_groups:
                    obj.vertex_groups.remove(obj.vertex_groups[vg_name])
            # Create new vertex groups
            vg_1 = obj.vertex_groups.new(name="+PC3")
            vg_2 = obj.vertex_groups.new(name="-PC3")
            vg_3 = obj.vertex_groups.new(name="+PC2")
            vg_4 = obj.vertex_groups.new(name="-PC2")
            vg_5 = obj.vertex_groups.new(name="+PC1")
            vg_6 = obj.vertex_groups.new(name="-PC1")
            
            # Assign vertices of +PC3 faces to +PC3 vertex group
            for face_idx in pc3_pos_faces:
                face = mesh.polygons[face_idx]
                for vert_idx in face.vertices:
                    vg_1.add([vert_idx], 1.0, "REPLACE")
            
            # Assign vertices of -PC3 faces to -PC3 vertex group
            for face_idx in pc3_neg_faces:
                face = mesh.polygons[face_idx]
                for vert_idx in face.vertices:
                    vg_2.add([vert_idx], 1.0, "REPLACE")

            # Assign vertices of +PC2 faces to +PC2 vertex group
            for face_idx in pc2_pos_faces:
                face = mesh.polygons[face_idx]
                for vert_idx in face.vertices:
                    vg_3.add([vert_idx], 1.0, "REPLACE")
            
            # Assign vertices of -PC2 faces to -PC2 vertex group
            for face_idx in pc2_neg_faces:
                face = mesh.polygons[face_idx]
                for vert_idx in face.vertices:
                    vg_4.add([vert_idx], 1.0, "REPLACE")
            
            # Assign vertices of +PC1 faces to +PC1 vertex group
            for face_idx in pc1_pos_faces:
                face = mesh.polygons[face_idx]
                for vert_idx in face.vertices:
                    vg_5.add([vert_idx], 1.0, "REPLACE")

            # Assign vertices of -PC1 faces to -PC1 vertex group
            for face_idx in pc1_neg_faces:
                face = mesh.polygons[face_idx]
                for vert_idx in face.vertices:
                    vg_6.add([vert_idx], 1.0, "REPLACE")

            # Step 7: Construct rotation matrix to align current orientation to target (X, Y, Z)
            x_axis = np.cross(y_axis, z_axis)  # Compute PC1 (orthogonal to PC2 and PC3)
            x_axis = x_axis / np.linalg.norm(x_axis)  # Normalize
            y_axis = np.cross(z_axis, x_axis)  # Recompute y_axis for right-handed system
            y_axis = y_axis / np.linalg.norm(y_axis)
            
            # Debug: Print current orientation basis
            self.report({'INFO'}, f"Current Orientation (x_axis, y_axis, z_axis):\n{x_axis}\n{y_axis}\n{z_axis}")

            # Current orientation matrix (columns are PC1, PC2, PC3)
            current_basis = Matrix([x_axis, y_axis, z_axis]).transposed()

            # Target orientation: align to world X, Y, Z
            target_basis = Matrix.Identity(3)  # [[1,0,0], [0,1,0], [0,0,1]]

            # Compute rotation matrix: current_basis * rotation = target_basis
            rotation_matrix = current_basis.inverted() @ target_basis
            
            # Convert to 4x4 for object transformation (no scaling or division)
            rotation_matrix = rotation_matrix.to_4x4()
            
            # Debug: Print rotation matrix and verify orthogonality
            self.report({'INFO'}, f"Rotation Matrix:\n{rotation_matrix}")
            det = rotation_matrix.determinant()
            self.report({'INFO'}, f"Rotation Matrix Determinant: {det:.6f} (should be ~1 for valid rotation)")
            is_orthogonal = np.allclose(np.dot(rotation_matrix.to_3x3(), rotation_matrix.to_3x3().transposed()), np.eye(3), atol=1e-6)
            self.report({'INFO'}, f"Rotation Matrix Orthogonal: {is_orthogonal}")

            # Apply only rotation (no translation)
            obj.matrix_world = rotation_matrix @ Matrix.Translation(obj.matrix_world.translation)
            bpy.ops.object.select_all(action='DESELECT')  # Deselect all objects
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj  # Set active object
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)


            # Step 8: Find vertices with minimum x, y, z coordinates and translate mesh
            vertices_world = [obj.matrix_world @ v.co for v in mesh.vertices]
            min_x_idx = min(range(len(vertices_world)), key=lambda i: vertices_world[i].x)
            min_y_idx = min(range(len(vertices_world)), key=lambda i: vertices_world[i].y)
            min_z_idx = min(range(len(vertices_world)), key=lambda i: vertices_world[i].z)
            
            min_x = vertices_world[min_x_idx].x
            min_y = vertices_world[min_y_idx].y
            min_z = vertices_world[min_z_idx].z
            
            # Debug: Print minimum vertices
            self.report({'INFO'}, f"Min X vertex (index {min_x_idx}): {vertices_world[min_x_idx]}")
            self.report({'INFO'}, f"Min Y vertex (index {min_y_idx}): {vertices_world[min_y_idx]}")
            self.report({'INFO'}, f"Min Z vertex (index {min_z_idx}): {vertices_world[min_z_idx]}")

            # Compute translation to move min vertices to x=0, y=0, z=0
            translation_vector = Vector((-min_x, -min_y, -min_z))
            
            # Apply translation
            obj.matrix_world = Matrix.Translation(translation_vector) @ obj.matrix_world
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
            
            # Debug: Verify new positions of minimum vertices
            vertices_world_after = [obj.matrix_world @ v.co for v in mesh.vertices]
            new_x = vertices_world_after[min_x_idx].x
            new_y = vertices_world_after[min_y_idx].y
            new_z = vertices_world_after[min_z_idx].z
            self.report({'INFO'}, f"New Min X vertex: x={new_x:.6f} (should be ~0)")
            self.report({'INFO'}, f"New Min Y vertex: y={new_y:.6f} (should be ~0)")
            self.report({'INFO'}, f"New Min Z vertex: z={new_z:.6f} (should be ~0)")

            self.report({'INFO'}, f"Aligned {obj.name} using PCA and translated to min vertices at (0,0,0)")

        elif context.scene.alignment_mode == '3-2-1':
            self.report({'ERROR'}, "3-2-1 alignment mode is not yet implemented")
            return {'CANCELLED'}

        self.report({'INFO'}, f"Alignment operation executed in {time.time() - t:.3f} seconds")
        return {'FINISHED'}

class WORKPIECE_OT_AlignmentInvert(Operator):
    bl_idname = "workpiece.alignment_invert"
    bl_label = "Invert Alignment Axis"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = "Invert the alignment axis of the workpiece"

    def execute(self, context):
        obj = bpy.context.active_object
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "No active mesh object found")
            return {'CANCELLED'}

        # Invert alignment axis
        t = time.time()  # start timer

        # Set origin: geometry to origin
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')

        invert_axis = None
        if context.scene.invert_X:
            obj.rotation_euler.x += math.pi  # Rotate 180 degrees around X axis
            invert_axis = 'X'
        if context.scene.invert_Y:
            obj.rotation_euler.y += math.pi  # Rotate 180 degrees around Y axis
            invert_axis = 'Y'
        if context.scene.invert_Z:
            obj.rotation_euler.z += math.pi  # Rotate 180 degrees around Z axis
            invert_axis = 'Z'
        if not invert_axis:
            self.report({'ERROR'}, "No axis selected for inversion")
            return {'CANCELLED'}
        
        # Apply the rotation
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
        
        self.report({'INFO'}, f"Alignment axis inverted along {invert_axis} in {time.time() - t:.3f} seconds")
        return {'FINISHED'}



class WORKPIECE_PT_MainPanel(Panel):
    bl_label = "Workpiece Processor"
    bl_idname = "WORKPIECE_PT_MainPanel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Workpiece"

    def draw(self, context):
        layout = self.layout
        layout.label(text="Step-by-Step Processing")
        layout.prop(context.scene, "stl_file")
        layout.operator("workpiece.import_stl", text="Import STL")
        layout.separator()
        layout.label(text="Canonical Alignment")
        layout.prop(context.scene, "alignment_mode", text="Mode")
        layout.operator("workpiece.alignment", text="Align Workpiece")
        layout.label(text="Rotate by 180° About Axis")
        layout.row(align=True)
        layout.prop(context.scene, "invert_X", text="X")
        layout.prop(context.scene, "invert_Y", text="Y")
        layout.prop(context.scene, "invert_Z", text="Z")
        layout.row(align=True)
        layout.operator("workpiece.alignment_invert", text="Invert axis", icon='ARROW_LEFTRIGHT')