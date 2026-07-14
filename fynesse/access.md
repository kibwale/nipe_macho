# `access.py` — XML-to-YOLO Dataset Converter

A small utility module that converts Pascal VOC–style XML annotations (RDD2020 format) into YOLO `.txt` label files, filtered to a target class list, plus a `data.yaml` for training.

---

## Functions

### `find_xml_dirs(root)`
**Purpose:** Finding XML directories.

Walks the entire dataset tree starting at `root` and returns every folder named `xmls` that it finds. This is what lets the module work across nested per-country folders (`train/Japan/annotations/xmls`, `train/Czech/annotations/xmls`, etc.) without hardcoding country names.

```python
xml_dirs = access.find_xml_dirs("/content/rdd2020")
# -> ['/content/rdd2020/train/Japan/annotations/xmls', ...]
```

---

### `convert_annotation(xml_path, txt_path, classes)`
**Purpose:** Converting individual annotations.

Takes a single XML file and converts it to one YOLO `.txt` file:

1. Reads image `width`/`height` from the XML's `<size>` tag.
2. Loops through every `<object>` in the file.
3. Keeps only objects whose `<name>` is in `classes` — everything else is dropped.
4. Converts the kept `<bndbox>` (xmin/ymin/xmax/ymax, pixel coordinates) into YOLO's normalized format: `class_id x_center y_center width height`, each value scaled 0–1.
5. Writes the result to `txt_path` (empty file if no matching objects were found).

This is called internally by `convert_dataset` — you generally don't need to call it directly, except for debugging a single file.

---

### `convert_dataset(root, classes)`
**Purpose:** Processing the entire dataset.

The main entry point. Orchestrates the whole pipeline:

1. Calls `find_xml_dirs(root)` to locate every annotation folder.
2. For each one, derives the sibling `images/` folder and creates a matching `labels/` folder.
3. Converts every XML file in that folder using `convert_annotation`.
4. Creates an **empty** `.txt` file for any image that has no matching annotations — YOLO expects every image to have a label file, even if it's a negative (background) example.
5. Tallies and prints total boxes converted and how many images contain at least one target box.

```python
classes = access.convert_dataset("/content/rdd2020", ["D40"])
```

Returns the `classes` list back (unchanged) — useful for passing straight into `write_data_yaml`.

---

### `write_data_yaml(root, classes, train_subdir="train/images", val_subdir="test/images")`
**Purpose:** Writing `data.yaml` for YOLO training.

Writes a `data.yaml` file at `root/data.yaml` containing the `train`/`val` image paths, number of classes (`nc`), and class names — the config file Ultralytics' `model.train(data=...)` expects.

```python
access.write_data_yaml("/content/rdd2020", classes)
```

---

## How it all fits together

```
convert_dataset(root, classes)
        │
        ├─→ find_xml_dirs(root)              # locate every annotations/xmls folder
        │
        └─→ for each xmls folder:
                └─→ convert_annotation(...)   # one XML file -> one YOLO .txt file
                        (called once per .xml file found)
```

`write_data_yaml` runs separately afterward, once conversion is done and you know the final class list.

---



### Notes
- `access.py` must be in your working directory (or on `sys.path`) for `import access` to work. If it lives elsewhere (e.g. Drive), add its folder first:
  ```python
  import sys
  sys.path.append("/content/drive/MyDrive/potholes")
  ```
- If you edit `access.py` after already importing it in the same session, Python caches the old version. Either restart the runtime, or reload it:
  ```python
  import importlib
  importlib.reload(access)
  ```
- To debug just one step instead of the full pipeline:
  ```python
  access.find_xml_dirs(DRIVE_ROOT)          # see which folders it found
  access.convert_annotation(xml_path, txt_path, ["D40"])  # convert a single file
  ```
