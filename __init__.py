import bpy
from .hard_straight import (
    WORKPIECE_OT_ImportSTL,
    WORKPIECE_OT_Alignment,
    WORKPIECE_PT_MainPanel
)

def register():
    bpy.utils.register_class(WORKPIECE_OT_ImportSTL)
    bpy.utils.register_class(WORKPIECE_OT_Alignment)
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
    bpy.types.Scene.angle_deviation_pc3 = bpy.props.FloatProperty(
        name="PC3 Angle Deviation",
        description="Allowed angle deviation between +Z face normals and PC3 for alignment (degrees)",
        default=10,
        min=0.0,
        max=90.0,
        precision=2
    )
    bpy.types.Scene.angle_deviation_pc2 = bpy.props.FloatProperty(
        name="PC2 Angle Deviation",
        description="Allowed angle deviation between +Y face normals and PC2 for alignment (degrees)",
        default=30,
        min=0.0,
        max=90.0,
        precision=2
    )
    bpy.types.Scene.angle_deviation_pc1 = bpy.props.FloatProperty(
        name="PC1 Angle Deviation",
        description="Allowed angle deviation between +X face normals and PC1 for alignment (degrees)",
        default=45,
        min=0.0,
        max=90.0,
        precision=2
    )

def unregister():
    bpy.utils.unregister_class(WORKPIECE_OT_ImportSTL)
    bpy.utils.unregister_class(WORKPIECE_OT_Alignment)
    bpy.utils.unregister_class(WORKPIECE_PT_MainPanel)
    
    # Unregister properties from the scene
    del bpy.types.Scene.stl_file
    del bpy.types.Scene.alignment_mode
    del bpy.types.Scene.angle_deviation_pc3
    del bpy.types.Scene.angle_deviation_pc2
    del bpy.types.Scene.angle_deviation_pc1

if __name__ == "__main__":
    register()