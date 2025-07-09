import bpy
from .hard_straight import (
    WORKPIECE_OT_ImportSTL,
    WORKPIECE_OT_Alignment,
    WORKPIECE_OT_AlignmentInvert,
    WORKPIECE_PT_MainPanel
)

def register():
    bpy.utils.register_class(WORKPIECE_OT_ImportSTL)
    bpy.utils.register_class(WORKPIECE_OT_Alignment)
    bpy.utils.register_class(WORKPIECE_OT_AlignmentInvert)
    bpy.utils.register_class(WORKPIECE_PT_MainPanel)
    
    # Register properties for the scene
    # These properties will be used to store user inputs for the workpiece processing
    bpy.types.Scene.stl_file = bpy.props.StringProperty(
        name="STL File",
        description="Path to a single STL file",
        subtype='FILE_PATH',
        default=""
    )
    bpy.types.Scene.alignment_mode = bpy.props.EnumProperty(
        name="Alignment Mode",
        description="Choose the alignment mode",
        items=[
            ('AUTO', "Automatic (PCA)", "Align using Principal Component Analysis"),
            ('3-2-1', "3-2-1 Point Alignment", "Align using user-defined points"),
        ],
        default='AUTO'
    )
    
    bpy.types.Scene.invert_X = bpy.props.BoolProperty(
        name="Invert X Axis",
        description="Invert the alignment along the X axis",
        default=False
    )
    bpy.types.Scene.invert_Y = bpy.props.BoolProperty(
        name="Invert Y Axis",
        description="Invert the alignment along the Y axis",
        default=False
    )
    bpy.types.Scene.invert_Z = bpy.props.BoolProperty(
        name="Invert Z Axis",
        description="Invert the alignment along the Z axis",
        default=False
    )

def unregister():
    bpy.utils.unregister_class(WORKPIECE_OT_ImportSTL)
    bpy.utils.unregister_class(WORKPIECE_OT_Alignment)
    bpy.utils.unregister_class(WORKPIECE_OT_AlignmentInvert)
    bpy.utils.unregister_class(WORKPIECE_PT_MainPanel)
    
    # Unregister properties from the scene
    del bpy.types.Scene.stl_file
    del bpy.types.Scene.alignment_mode
    del bpy.types.Scene.invert_X
    del bpy.types.Scene.invert_Y
    del bpy.types.Scene.invert_Z

if __name__ == "__main__":
    register()