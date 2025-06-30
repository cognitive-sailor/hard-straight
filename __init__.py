import bpy
from .hard_straight import (
    WORKPIECE_OT_ImportSTL,
    WORKPIECE_PT_MainPanel
)

def register():
    bpy.utils.register_class(WORKPIECE_OT_ImportSTL)
    bpy.utils.register_class(WORKPIECE_PT_MainPanel)
    bpy.types.Scene.stl_file = bpy.props.StringProperty(
        name="STL File",
        description="Path to a single STL file",
        subtype='FILE_PATH',
        default=""
    )

def unregister():
    bpy.utils.unregister_class(WORKPIECE_OT_ImportSTL)
    bpy.utils.unregister_class(WORKPIECE_PT_MainPanel)
    del bpy.types.Scene.stl_file

if __name__ == "__main__":
    register()