# Rendu d'un .vdb exporté par LiquidVM 3D (touche E), en headless :
#   blender -b --factory-startup --python tools/blender_render_vdb.py -- <fichier.vdb> <sortie.png>
# Ou à ouvrir dans l'onglet Scripting de Blender et lancer avec ▶ (adapter PATH/OUT).
#
# Ce que le script fait — et qu'il faut refaire à la main sinon :
# 1. PURGE de la scène par défaut : le cube 2×2×2 vit dans une collection enfant
#    et remplit le cadre si on l'oublie (rendu tout gris).
# 2. Import du volume : dans l'interface c'est Add → Volume → Import OpenVDB
#    (PAS File → Import — il n'y a pas d'entrée VDB là-bas !).
# 3. Matériau : Principled Volume branché sur la prise VOLUME du Material Output
#    (pas Surface), Density ≈ 60–120 (nos densités valent ~0–3 dans une boîte de 1 m).
# 4. Une lumière Soleil + une caméra qui vise le volume, moteur Cycles.
import math
import sys

import bpy

if "--" in sys.argv:
    argv = sys.argv[sys.argv.index("--") + 1 :]
    PATH, OUT = argv[0], argv[1]
else:
    PATH = r"C:\chemin\vers\liquidvm-XXXX.vdb"  # à adapter si lancé depuis l'éditeur
    OUT = r"C:\chemin\vers\rendu.png"

sc = bpy.context.scene

# 1. Vider TOUTE la scène par défaut (toutes collections confondues).
for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)
sc.world.use_nodes = True
sc.world.node_tree.nodes["Background"].inputs[0].default_value = (0, 0, 0, 1)

# 2. Import du volume.
bpy.ops.object.volume_import(filepath=PATH, align="WORLD")
vol_obj = bpy.context.object

# 3. Matériau : fumée (density) + lueur corps noir (temperature).
mat = bpy.data.materials.new("liquidvm-volume")
mat.use_nodes = True
nt = mat.node_tree
nt.nodes.clear()
pv = nt.nodes.new("ShaderNodeVolumePrincipled")
pv.inputs["Density"].default_value = 90.0
pv.inputs["Blackbody Intensity"].default_value = 1.0
out_node = nt.nodes.new("ShaderNodeOutputMaterial")
nt.links.new(pv.outputs["Volume"], out_node.inputs["Volume"])
vol_obj.data.materials.append(mat)

# 4. Caméra contrainte à viser le volume + soleil.
cam_obj = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
sc.collection.objects.link(cam_obj)
cam_obj.location = (1.6, -1.6, 0.35)
track = cam_obj.constraints.new("TRACK_TO")
track.target = vol_obj
track.track_axis = "TRACK_NEGATIVE_Z"
track.up_axis = "UP_Y"
sc.camera = cam_obj

sun = bpy.data.lights.new("sun", "SUN")
sun.energy = 3.0
sun_obj = bpy.data.objects.new("sun", sun)
sc.collection.objects.link(sun_obj)
sun_obj.rotation_euler = (0.9, 0.2, 0.6)

sc.render.engine = "CYCLES"
sc.cycles.samples = 64
sc.render.resolution_x = 1280
sc.render.resolution_y = 960
sc.render.filepath = OUT
bpy.ops.render.render(write_still=True)
print("RENDER-OK:", OUT)
