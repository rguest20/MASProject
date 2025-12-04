class VirtualFS:
    """
    Lightweight file + canvas system.
    """
    def __init__(self, name="world", width=32, height=32):
        self.name = name
        self.files = {}
        self.canvas_w = width
        self.canvas_h = height
        self.canvas = [[(255,255,255) for _ in range(width)]
                       for __ in range(height)]

    # text operations
    def read_text(self, path):
        return self.files.get(path, "")

    def write_text(self, path, text):
        self.files[path] = str(text)

    def append_text(self, path, text):
        self.files[path] = self.files.get(path, "") + str(text)

    def list_paths(self, prefix="/"):
        prefix = prefix if prefix.endswith("/") else prefix + "/"
        return [p for p in self.files.keys() if p.startswith(prefix)]

    # canvas operations
    def draw_pixel(self, x, y, rgb):
        if 0 <= x < self.canvas_w and 0 <= y < self.canvas_h:
            r,g,b = rgb
            self.canvas[y][x] = (
                max(0,min(255,int(r))),
                max(0,min(255,int(g))),
                max(0,min(255,int(b)))
            )
