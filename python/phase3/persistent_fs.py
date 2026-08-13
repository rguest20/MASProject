import os, json

try:
    from PIL import Image  # type: ignore
except Exception:
    Image = None

class PersistentFS:
    """
    Safe, bounded, on-disk virtual filesystem.
    Paths map to real files inside a sandbox root.
    Provides:
      - read/write/append text
      - read/write JSON
      - PNG canvas support
      - dynamic path listing using os.walk
    """
    def __init__(self, root):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)
        self.canvas_w = 0
        self.canvas_h = 0
        self.canvas = None

    # ------------------------
    # Path safety
    # ------------------------
    def _resolve(self, path):
        # Clean leading slash
        clean = path.lstrip("/")
        full = os.path.abspath(os.path.join(self.root, clean))

        # Prevent ../ escape attacks
        if not full.startswith(self.root):
            raise PermissionError("Illegal path escape attempt.")
        return full

    def make_dirs(self, path):
        if not path or path in ("/", ""):
            return
        full = self._resolve(path)
        os.makedirs(full, exist_ok=True)

    # ------------------------
    # Path listing (critical)
    # ------------------------
    def list_paths(self, prefix="/"):
        """
        Returns all virtual paths starting with prefix.
        We compute this by scanning the real directory structure.
        """
        # Normalize prefix
        if prefix is None:
            prefix = "/"
        if not prefix.startswith("/"):
            prefix = "/" + prefix

        results = []

        # Walk the actual directory tree
        for dirpath, dirs, files in os.walk(self.root):
            for f in files:
                full = os.path.join(dirpath, f)
                # Compute virtual path: remove root prefix and add leading slash
                vpath = "/" + os.path.relpath(full, self.root).replace("\\", "/")
                if vpath.startswith(prefix) or prefix == "/":
                    results.append(vpath)

        return results

    # ------------------------
    # Text I/O
    # ------------------------
    def read_text(self, path):
        full = self._resolve(path)
        if not os.path.exists(full):
            return ""
        with open(full, "r", encoding="utf-8") as f:
            return f.read()

    def write_text(self, path, text):
        full = self._resolve(path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(str(text))

    def append_text(self, path, text):
        full = self._resolve(path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "a", encoding="utf-8") as f:
            f.write(str(text))

    # ------------------------
    # JSON
    # ------------------------
    def put_json(self, path, obj):
        full = self._resolve(path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            json.dump(obj, f)

    def read_json(self, path):
        full = self._resolve(path)
        if not os.path.exists(full):
            return None
        with open(full, "r", encoding="utf-8") as f:
            return json.load(f)

    # ------------------------
    # Canvas
    # ------------------------
    def draw_pixel(self, x, y, rgb):
        if self.canvas is None or self.canvas_w <= 0 or self.canvas_h <= 0:
            self.create_canvas(32, 32)
        if not (0 <= x < self.canvas_w and 0 <= y < self.canvas_h):
            return
        r, g, b = rgb
        self.canvas[y][x] = (
            max(0, min(255, int(r))),
            max(0, min(255, int(g))),
            max(0, min(255, int(b))),
        )

    def save_canvas(self, path, pixel_grid):
        if Image is None:
            return
        full = self._resolve(path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        h = len(pixel_grid)
        w = len(pixel_grid[0])
        img = Image.new("RGB", (w, h))
        flat = []
        for row in pixel_grid:
            flat.extend(row)
        img.putdata(flat)
        img.save(full)

    def create_canvas(self, width, height):
        self.canvas_w = int(width)
        self.canvas_h = int(height)
        self.canvas = [[(255, 255, 255) for _ in range(self.canvas_w)] for _ in range(self.canvas_h)]
        return self.canvas
