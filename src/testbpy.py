import bpy

bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 0))

cube_obj = bpy.context.object

loc = cube_obj.location

loc.x = 5
loc.y = 5
loc.z = 5