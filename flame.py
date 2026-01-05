'''
Make 2 flame graphs to the same scale for comparison
of UKCA timings with different photolysis schemes.
Times given for one normal photolysis timestep in milliseconds.
'''

import matplotlib.pyplot as plt


def plot_flame(layer_0, layer_1, layer_2, ukca_max, scheme):
  fig, ax = plt.subplots(figsize=(14,3))

  # Layer 0 - UKCA.
  time = layer_0[2]
  ax.barh(y=0, width=time, left=0, height=1, color=layer_0[3], edgecolor="black")
  ax.text(time / 2, 0, layer_0[0], ha="center", va="center", fontsize=12)
  
  # Layer 1 - Main routines.
  for name, start, time, colour in layer_1:
    ax.barh(y=1, width=time, left=start, height=1, color=colour, edgecolor="black")
    if time > 20:
      ax.text(start + time / 2, 1, name, ha="center", va="center", fontsize=12)
    elif time > 5:
      ax.text(start + time / 2, 0.3, name, ha="left", va="center", fontsize=12)

  # Layer 2 - Photolysis scheme.
  start = layer_2[1]
  time = layer_2[2]
  ax.barh(y=2, width=time, left=start, height=1, color=layer_2[3], edgecolor="black")
  if time > 20:
    ax.text(start + time / 2, 2, layer_2[0], ha="center", va="center", fontsize=12)
  elif time > 5:
    ax.text(start + time / 2, 2.7, layer_2[0], ha="left", va="center", fontsize=12)
  
  # Formatting.
  ax.set_xlim(0, ukca_max+1)
  ax.set_xlabel("Real time / ms")
  ax.set_title(f"Real time spent on main UKCA routines on a photolysis timestep, using {scheme}")
  ax.yaxis.set_visible(False)
  ax.spines["top"].set_visible(False)
  ax.spines["left"].set_visible(False)
  ax.spines["right"].set_visible(False)
  plt.tight_layout()
  plt.show()
  

# Longest UKCA can take.
ukca_max = 309.24

# Fast-JX timings - name, start time, time taken, colour.

# How long UKCA takes.
fj_layer_0 = ("UKCA", 0, 309.24, "yellow")

# All main routines in UKCA.
fj_layer_1 = [
 ("Data", 0.14, 1.8, "gold"),
 ("Humidity", 1.94, 4.8444, "gold"),
 ("Photolysis control", 6.7844, 51.73, "orange"),
 ("Main chemsitry reactions", 58.5144, 205, "gold"),
 ("Data", 263.5144, 1.5032, "gold"),
 ("Plumes", 265.0176, 2.7727, "gold"),
 ("STASH", 267.7903, 41.5, "gold")
]

# Photolysis scheme.
fj_layer_2 = ("Fast-JX photolysis", 12.2844, 46.189, "red")

# Random forest timings - name, start time, time taken, colour.

# How long UKCA takes.
ml_layer_0 = ("UKCA", 0, 269.5503, "yellow")

# All main routines in UKCA.
ml_layer_1 = [
 ("Data", 0.14, 1.8, "gold"),
 ("Humidity", 1.94, 4.8444, "gold"),
 ("Photolysis control", 6.7844, 11.559, "orange"),
 ("Main chemsitry reactions", 18.3744, 205, "gold"),
 ("Data", 223.3744, 1.5032, "gold"),
 ("Plumes", 224.8776, 2.7727, "gold"),
 ("STASH", 227.6503, 41.9, "gold")
]

# Photolysis scheme.
ml_layer_2 = ("Random forest photolysis", 11.8359, 6.5075, "red")

plot_flame(fj_layer_0, fj_layer_1, fj_layer_2, ukca_max, "Fast-JX")
plot_flame(ml_layer_0, ml_layer_1, ml_layer_2, ukca_max, "random forest")
