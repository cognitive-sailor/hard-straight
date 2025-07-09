import bpy
import numpy as np
import scipy
import os
import mathutils
from mathutils import Vector, Matrix
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

    def execute(self, context):
        obj = bpy.context.active_object
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "No active mesh object found")
            return {'CANCELLED'}

        # Perform canonical alignment
        t = time.time()  # start timer

        mesh = obj.data
        bpy.ops.object.mode_set(mode='OBJECT')  # Ensure we are in object mode

        # Placeholder for alignment logic
        if context.scene.alignment_mode == 'AUTO':
            # Automatic alignment using PCA

            # Set origin: geometry to origin
            bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')

            # Get vertex coordinates
            vertices = np.array([v.co for v in obj.data.vertices])
            if len(vertices) < 3:
                self.report({'ERROR'}, "Not enough vertices for PCA alignment")
                return {'CANCELLED'}

            # Perform PCA
            mean = np.mean(vertices, axis=0) # Calculate mean
            centered = vertices - mean # Center the mesh
            cov_matrix = np.cov(centered.T) # Calculate covariance matrix
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix) # Calculate eigenvalues and eigenvectors
            order = np.argsort(eigenvalues)[::-1] # Sort eigenvalues in descending order
            eigenvectors = eigenvectors[:, order] # Sort eigenvectors accordingly

            if context.scene.invert_X:
                eigenvectors[:, 0] *= -1  # Invert X axis
            if context.scene.invert_Y:
                eigenvectors[:, 1] *= -1  # Invert Y axis
            if context.scene.invert_Z:
                eigenvectors[:, 2] *= -1  # Invert Z axis

            # Create rotation matrix to align principal axes with world axes
            rotation_matrix = mathutils.Matrix(eigenvectors.T).to_4x4()
            obj.matrix_world = rotation_matrix  # Apply rotation

            # Move the object to the origin
            obj.location.x = obj.location.x + abs(min(vertices[:, 0]))
            obj.location.y = obj.location.y + abs(min(vertices[:, 1]))
            obj.location.z = obj.location.z + abs(min(vertices[:, 2]))

            bpy.ops.object.transform_apply(location=True, rotation=True, scale=False)  # Apply transformation

            # # Set origin to 3D cursor
            # bpy.context.scene.cursor.location = Vector((0, 0, 0))  # Set cursor to origin
            # bpy.ops.object.origin_set(type='ORIGIN_CURSOR') 
            self.report({'INFO'}, f"Aligned {obj.name} using PCA")

        elif context.scene.alignment_mode == '3-2-1':
            # 3-2-1 alignment using user-defined points
            # This is a placeholder for the actual implementation
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