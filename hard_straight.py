import bpy
import numpy as np
import scipy
import os
from pathlib import Path
from bpy.types import Operator, Panel
import time

class WORKPIECE_OT_ImportSTL(Operator):
    bl_idname = "workpiece.import_stl"
    bl_label = "Import STL"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
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
