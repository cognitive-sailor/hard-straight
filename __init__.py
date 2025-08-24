import bpy
from .hard_straight import (
    WORKPIECE_OT_ImportSTL,
    WORKPIECE_OT_Alignment,
    WORKPIECE_OT_Flatness,
    WORKPIECE_OT_Batch,
    WORKPIECE_PT_MainPanel
)
from .strike_planner import (
    STRIKEGEN_PT_StrikeSettings,
    STRIKEGEN_PT_StrikePropertyGroup,
    STRIKEGEN_OT_GenerateStrikes,
    STRIKEGEN_OT_WriteStrikesCSV,
    STRIKEGEN_PT_MainPanel
)

def register():
    bpy.utils.register_class(WORKPIECE_OT_ImportSTL)
    bpy.utils.register_class(WORKPIECE_OT_Alignment)
    bpy.utils.register_class(WORKPIECE_OT_Flatness)
    bpy.utils.register_class(WORKPIECE_OT_Batch)
    bpy.utils.register_class(WORKPIECE_PT_MainPanel)
    bpy.utils.register_class(STRIKEGEN_PT_MainPanel)
    bpy.utils.register_class(STRIKEGEN_PT_StrikeSettings)
    bpy.utils.register_class(STRIKEGEN_PT_StrikePropertyGroup)
    bpy.utils.register_class(STRIKEGEN_OT_GenerateStrikes)
    bpy.utils.register_class(STRIKEGEN_OT_WriteStrikesCSV)
    
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
    bpy.types.Scene.flatness_top = bpy.props.BoolProperty(
        name="Top",
        description="Calculate flatness for top vertex group",
        default=True
    )
    bpy.types.Scene.flatness_bottom = bpy.props.BoolProperty(
        name="Bottom",
        description="Calculate flatness for bottom vertex group",
        default=True
    )
    bpy.types.Scene.flatness_front = bpy.props.BoolProperty(
        name="Front",
        description="Calculate flatness for front vertex group",
        default=True
    )
    bpy.types.Scene.flatness_back = bpy.props.BoolProperty(
        name="Back",
        description="Calculate flatness for back vertex group",
        default=True
    )
    bpy.types.Scene.flatness_right = bpy.props.BoolProperty(
        name="Right",
        description="Calculate flatness for right vertex group",
        default=True
    )
    bpy.types.Scene.flatness_left = bpy.props.BoolProperty(
        name="Left",
        description="Calculate flatness for left vertex group",
        default=True
    )
    bpy.types.Scene.flatness_max_vertices = bpy.props.IntProperty(
        name="Max Vertices for Flatness",
        description="Maximum number of vertices to sample for flatness calculation (0 for all)",
        default=0,
        min=0
    )
    bpy.types.Scene.stl_mesh = bpy.props.PointerProperty(type=bpy.types.Object, name='stl_mesh', description='The main .STL mesh for processing')

    # Batch processing
    bpy.types.Scene.stl_directory = bpy.props.StringProperty(
        name="STL Directory",
        description="Directory containing STL files",
        subtype='DIR_PATH',
        default=""
    )
    bpy.types.Scene.canonical_alignment = bpy.props.BoolProperty(
        name="Canonical Alignment",
        description="Include Canonical Alignment?",
        default=True
    )
    bpy.types.Scene.flatness_calculation = bpy.props.BoolProperty(
        name="Flatness Calculation",
        description="Include Flatness Calculation?",
        default=True
    )

    # Strike Generator
    bpy.types.Scene.StrikeSettings = bpy.props.PointerProperty(type=STRIKEGEN_PT_StrikeSettings)
    bpy.types.Mesh.Strikes = bpy.props.CollectionProperty(type=STRIKEGEN_PT_StrikePropertyGroup)

def unregister():
    bpy.utils.unregister_class(WORKPIECE_OT_ImportSTL)
    bpy.utils.unregister_class(WORKPIECE_OT_Alignment)
    bpy.utils.unregister_class(WORKPIECE_OT_Flatness)
    bpy.utils.unregister_class(WORKPIECE_OT_Batch)
    bpy.utils.unregister_class(WORKPIECE_PT_MainPanel)
    bpy.utils.unregister_class(STRIKEGEN_PT_MainPanel)
    bpy.utils.unregister_class(STRIKEGEN_PT_StrikeSettings)
    bpy.utils.unregister_class(STRIKEGEN_PT_StrikePropertyGroup)
    bpy.utils.unregister_class(STRIKEGEN_OT_GenerateStrikes)
    bpy.utils.unregister_class(STRIKEGEN_OT_WriteStrikesCSV)
    
    # Unregister properties from the scene
    del bpy.types.Scene.stl_file
    del bpy.types.Scene.alignment_mode
    del bpy.types.Scene.angle_deviation_pc3
    del bpy.types.Scene.angle_deviation_pc2
    del bpy.types.Scene.angle_deviation_pc1
    del bpy.types.Scene.flatness_top
    del bpy.types.Scene.flatness_bottom
    del bpy.types.Scene.flatness_front
    del bpy.types.Scene.flatness_back
    del bpy.types.Scene.flatness_right
    del bpy.types.Scene.flatness_left
    del bpy.types.Scene.flatness_max_vertices
    del bpy.types.Scene.stl_mesh

    del bpy.types.Scene.stl_directory
    del bpy.types.Scene.canonical_alignment
    del bpy.types.Scene.flatness_calculation

    del bpy.types.Scene.StrikeSettings
    del bpy.types.Mesh.Strikes

if __name__ == "__main__":
    register()