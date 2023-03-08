import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

safe_map   = lambda f, xs: list(map(f, xs))
safe_unzip = lambda xys: list(zip(*xys))

def safe_filter(f, xs):
  ret = []
  for x in xs:
    if f(x):
      ret.append(x)
  return ret

fst = lambda x: x[0]
snd = lambda x: x[1]

is_not_none = lambda x: x is not None

def barplot(info, filename = "f.png", color_each=True, xlabel="Time"):
  """
  info: list of (start: int, end: int , label: str, type: str) items

  labels map to which row, type map to which color.
  """
  assert len(info) > 0

  all_labels = sorted(set(i[2] for i in info))
  to_label_idx = {which:idx for idx,which in enumerate(all_labels)}

  all_types = sorted(set(i[3] for i in info))
  _colors = list(mcolors.TABLEAU_COLORS.keys())
  _to_color_idx = {which:idx for idx,which in enumerate(all_types)}
  def to_color_idx(which):
    return _to_color_idx[which] % len(_colors)

  min_start = min(i[0] for i in info)
  #min_width = 0.001*(max(i[1] for i in info) - min_start)
  #min_width = 0.005*(max(i[1] for i in info) - min_start)
  min_width = 0

  if not color_each:
    info = [(start-min_start,
             max(min_width, end-start),
             to_label_idx[label],
             _colors[to_color_idx(ttype)]) for
              start, end, label, ttype in info]
  else:
    info = [(start-min_start,
             max(min_width, end-start),
             to_label_idx[label],
             _colors[idx % len(_colors)]) for
              idx, (start, end, label, ttype) in enumerate(info)]

  starts, duration, row, colors = safe_unzip(info)

  plt.barh(row, duration, left=starts, color = colors)
  plt.xlabel(xlabel)
  plt.ylabel("Operation")
  plt.yticks(ticks = range(len(all_labels)), labels = all_labels)

  figure = plt.gcf() # get current figure
  figure.set_size_inches(12,6)
  plt.savefig(filename, dpi=400)
  plt.close()

LOGFILE = sys.argv[1]
with open(LOGFILE, "r") as f:
  def fix(line):
    try:
      x,y,z = line[:-1].split(",")
      return (int(x),int(y),z,"")
    except ValueError:
      return None
  info = safe_filter(is_not_none, map(fix, f.readlines()))
  barplot(info)
