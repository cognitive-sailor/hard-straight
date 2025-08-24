import bpy
import numpy as np
from bpy.types import Operator, Panel, PropertyGroup
import math
import random
import csv
import os

class STRIKEGEN_PT_StrikeSettings(PropertyGroup):
    strike_length: bpy.props.IntProperty(default = 14, min=1, max=30)
    strike_width: bpy.props.IntProperty(default = 2, min=1, max=10)
    x_edge_padding: bpy.props.IntProperty(default=1, min=0, max=10)
    y_edge_padding: bpy.props.IntProperty(default=1, min=0, max=10)
    inter_distance: bpy.props.FloatProperty(default=1, min=0, max=10)
    number_of_strikes: bpy.props.IntProperty(default=56, min=0, max=100)
    strike_export_dir: bpy.props.StringProperty(
        name="Export Dir",
        description="Select destination directory for CSV file containing strike data",
        subtype="DIR_PATH",
        default=""
    )
    process_all_files: bpy.props.BoolProperty(default=False)

# Define the custom Strike item as a PropertyGroup
class STRIKEGEN_PT_StrikePropertyGroup(PropertyGroup):
    ID: bpy.props.IntProperty(
        name="ID",
        description="Strike ID number, sequential strike.",
        default=0
    )
    x_location: bpy.props.IntProperty(
        name="X Location",
        description="X coordinate of the strike (in local mesh space)",
        default=0,
        min=0,
        max=300
    )
    y_location: bpy.props.IntProperty(
        name="Y Location",
        description="Y coordinate of the strike",
        default=0,
        min=0,
        max=40
    )
    z_location: bpy.props.FloatProperty(
        name="Z Location",
        description="Z coordinate of the strike",
        default=0,
        min=0,
        max=5.2
    )
    orientation: bpy.props.IntProperty(
        name="Orientation",
        description="Strike orientation angle (in degrees)",
        default=0,
        min=0,
        max=180
    )

class STRIKEGEN_OT_GenerateStrikes(Operator):
    bl_idname = "workpiece.generate_strikes"  # Unique ID (prefix with 'mesh.' for convention)
    bl_label = "Generate a series of strikes"
    bl_description = "Generates a series of non-overlapping set of strikes on top and bottom surface of the workpiece"
    bl_options = {'REGISTER', 'UNDO'}  # Enables undo/redo

    def execute(self, context):
        obj = context.active_object
        if obj is None or obj.type != 'MESH':
            self.report({'ERROR'}, "No active mesh object selected!")
            return {'CANCELLED'}
        # Move the workpiece to the Collection, if it's not already there
        if bpy.data.collections.get('Collection')==None: # If not, create a collection "Collection"
            bpy.ops.object.move_to_collection(collection_index=0, is_new=True, new_collection_name="Collection")
        else:
            bpy.ops.object.move_to_collection(collection_index=1)
        
        # Create a new collection "Strikes" for strike objects
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(False)
        if bpy.data.collections.get('Strikes')==None: # If not, create a collection "Strikes"
            bpy.ops.collection.create(name="Strikes")
            strikes_collection = bpy.data.collections["Strikes"]
            bpy.context.scene.collection.children.link(strikes_collection) # link collection to the Scene Collection
        else:
            # Remove old "Strikes" collection and all of its objects
            for strike_object in bpy.data.collections.get('Strikes').objects[:]:
                bpy.data.objects.remove(strike_object, do_unlink=True)
            bpy.data.collections.remove(bpy.data.collections["Strikes"])
            # Create a new collection "Strikes" for strike objects
            bpy.ops.collection.create(name="Strikes")
            strikes_collection = bpy.data.collections["Strikes"]
            bpy.context.scene.collection.children.link(strikes_collection) # link collection to the Scene Collection

        mesh = obj.data  # Access the mesh data block
        strikes = mesh.Strikes
        strike_settings = bpy.data.scenes['Scene'].StrikeSettings
        
        # Clear existing strikes (optional; comment out if you want to append)
        strikes.clear()

        # Adjust limits based on boundary padding
        x_min = strike_settings.x_edge_padding + strike_settings.strike_length/2
        x_max = 300 - (strike_settings.x_edge_padding + strike_settings.strike_length/2)
        y_min = strike_settings.y_edge_padding + strike_settings.strike_length/2
        y_max = 40 - (strike_settings.y_edge_padding + strike_settings.strike_length/2)

        attempts = 0
        max_attempts = 1e5  # Prevent infinite loops; adjust if needed

        for _ in range(strike_settings.number_of_strikes):
            strike = strikes.add()  # Adds a new StrikePropertyGroup item
            strike.ID = len(strikes) # Construct a new ID

            while attempts < max_attempts:
                self.x_location = round(random.uniform(x_min, x_max),0)
                self.y_location = round(random.uniform(y_min, y_max),0)
                self.z_location = random.choice([0, 5.2])
                self.orientation = round(random.uniform(0, 180),0)

                if self._check_no_overlap(mesh, strike_settings.inter_distance, strike_settings.strike_length,strike_settings.strike_width):
                    break
                attempts += 1

            if attempts == max_attempts:
                self.report({'ERROR'}, f"Could not find a non-overlapping position after maximum attempts.")
            else:
                # Save the strike
                strike.name = f"Strike_{strike.ID}"
                strike.x_location = int(self.x_location)
                strike.y_location = int(self.y_location)
                strike.z_location = self.z_location
                strike.orientation = int(self.orientation)
                
                bpy.data.scenes["Scene"].cursor.location[0] = self.x_location
                bpy.data.scenes["Scene"].cursor.location[1] = self.y_location
                bpy.data.scenes["Scene"].cursor.location[2] = self.z_location
                width = strike_settings.strike_width/strike_settings.strike_length
                bpy.ops.mesh.primitive_ico_sphere_add(
                    radius=strike_settings.strike_length/2, 
                    scale=(1, width, 0.2)
                    )
                strike_obj = bpy.context.selected_objects[0]
                strike_obj.name = f"Strike_{strike.ID}"
                strike_obj.rotation_euler[2] = strike.orientation*np.pi/180 # rotation angle in radians
                bpy.ops.object.move_to_collection(collection_index=2)
            # Force viewport redraw
            for area in context.window.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()

        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.data.scenes["Scene"].cursor.location[0] = 0
        bpy.data.scenes["Scene"].cursor.location[1] = 0
        bpy.data.scenes["Scene"].cursor.location[2] = 0
        self.report({'INFO'}, f"{strike_settings.number_of_strikes} successfully generated!")
        return {'FINISHED'}

    def _check_no_overlap(self, mesh, min_distance_padding, strike_length, strike_width):
        """
        Checks if this strike overlaps with existing strikes (approximating as circles).
        
        Returns:
            bool: True if no overlap, False otherwise.
        """
        strikes = mesh.Strikes
        # Approximate radius: max dimension / 2 + half padding
        radius = max(strike_length, strike_width) / 2 + min_distance_padding / 2
        for other in strikes:
            dx = self.x_location - other.x_location
            dy = self.y_location - other.y_location
            dz = self.z_location - other.z_location
            if abs(dz) > 4:
                continue
            distance = math.sqrt(dx**2 + dy**2)
            if distance < 2 * radius:  # Check in x-y plane (ignore z)
                return False
        return True

class STRIKEGEN_OT_WriteStrikesCSV(Operator):
    bl_idname = "workpiece.write_strikes_csv"
    bl_label = "Write strikes to a CSV"
    bl_description = "Writes all strikes that were generated for some workpiece to a CSV file"
    bl_options = {'REGISTER', 'UNDO'}  # Enables undo/redo

    def execute(self, context):
        export_file = bpy.data.scenes['Scene'].StrikeSettings.strike_export_dir + f"generated_strikes.csv"
        if context.scene.StrikeSettings.process_all_files:
            input_directory = context.scene.stl_directory
            self.report({'INFO'}, f"Generating and saving all files from {input_directory} to the {export_file}")
            stl_files = [f for f in os.listdir(input_directory) if f.lower().endswith('.stl')]
            for file in stl_files:
                # 1. Import a STL file
                bpy.ops.object.select_all(action='SELECT')
                bpy.ops.object.delete()
                context.scene.stl_file = os.path.join(input_directory,file) # set a file to inport
                bpy.ops.workpiece.import_stl()
                # 2. Canonical Alignment - AUTO
                bpy.ops.workpiece.alignment()
                # 3. Strike generation
                bpy.ops.workpiece.generate_strikes()
                # 4. Save strike data to a .csv file
                obj = context.active_object
                if obj is None or obj.type != 'MESH':
                    self.report({'ERROR'}, "No active mesh object selected!")
                    return {'CANCELLED'}
                self.report({'INFO'}, f"{obj.name}: Writing strike data to CSV file...")
                # Collect all strikes and their properties
                strike_data = []
                for strike in obj.data.Strikes[:]:
                    strike_data.append([obj.name, strike.ID, strike.x_location, strike.y_location, strike.z_location, strike.orientation])
                # Open a CSV file and write to it
                my_mode = "a" if os.path.exists(export_file) else "w"
                with open(export_file, mode=my_mode, newline="") as file:
                    writer = csv.writer(file)
                    if my_mode != "a":
                        writer.writerow(['workpiece', 'ID', 'X', 'Y', 'Z', 'alpha']) # write this line only at file creation
                    writer.writerows(strike_data)
                self.report({'INFO'}, f"Strike data saved to: {export_file}")
                # Force viewport redraw
                for area in context.window.screen.areas:
                    if area.type == 'VIEW_3D':
                        area.tag_redraw()
        else:
            obj = context.active_object
            if obj is None or obj.type != 'MESH':
                self.report({'ERROR'}, "No active mesh object selected!")
                return {'CANCELLED'}
            self.report({'INFO'}, f"{obj.name}: Writing strike data to CSV file...")
            # Collect all strikes and their properties
            strike_data = []
            for strike in obj.data.Strikes[:]:
                strike_data.append([obj.name, strike.ID, strike.x_location, strike.y_location, strike.z_location, strike.orientation])
            # Open a CSV file and write to it
            my_mode = "a" if os.path.exists(export_file) else "w"
            with open(export_file, mode=my_mode, newline="") as file:
                writer = csv.writer(file)
                if my_mode != "a":
                    writer.writerow(['workpiece', 'ID', 'X', 'Y', 'Z', 'alpha']) # write this line only at file creation
                writer.writerows(strike_data)
            self.report({'INFO'}, f"Strike data saved to: {export_file}")
            return {'FINISHED'}


class STRIKEGEN_PT_MainPanel(Panel):
    bl_label = "Strike Generator"
    bl_idname = "STRIKEGEN_PT_MainPanel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Workpiece"

    
    def draw(self, context):
        layout = self.layout
        strike_settings = bpy.data.scenes['Scene'].StrikeSettings
        layout.label(text="Strike Settings")
        box = layout.box()
        box.prop(strike_settings, "strike_length")
        box.prop(strike_settings, "strike_width")
        box.prop(strike_settings, "x_edge_padding")
        box.prop(strike_settings, "y_edge_padding")
        box.prop(strike_settings, "inter_distance")
        box.prop(strike_settings, "number_of_strikes")
        layout.separator()
        layout.label(text="Generate Strikes")
        layout.operator("workpiece.generate_strikes", text="Generate")
        layout.label(text="Write Strikes to a CSV file")
        box2 = layout.box()
        row = box2.row()
        row.prop(context.scene, "stl_directory")
        row.prop(strike_settings, "process_all_files", text="All files")
        box2.prop(strike_settings, "strike_export_dir")
        layout.operator("workpiece.write_strikes_csv", text="Write CSV")

