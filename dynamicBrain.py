import random, math
import numpy as np
import pygame   
import torch                               # sadece draw() için

# Renkler –  settings.py içinden kullanıyorsanız buradakileri silebilirsiniz
BLUE  = ( 32, 110, 215)
RED   = (222,  58,  52)
BLACK = (  0,   0,   0)

# Kenar: (from_id, to_id, weight, enabled)
Edge = tuple[int, int, float, bool]

class DynamicBrain:
    """
    Bias nöronu + etkin/pasif kenarlar + topolojik sıra içeren
    NEAT‑vari beyin.
    """
    BIAS_ID = -1                      # sabit 1.0

    # --------------------------- YAPILANDIRICI ------------------------- #
    def __init__(self,
                 input_size:  int,
                 output_size: int = 2,
                 mutation_rate: float = 0.05):

        self.input_size    = input_size
        self.output_size   = output_size
        self.mutation_rate = mutation_rate

        # 0 … input_size-1  +  bias
        self.input_neurons  = [DynamicBrain.BIAS_ID] + list(range(input_size))
        self.output_neurons = list(range(100, 100 + output_size))
        self.hidden_neurons: list[int] = []
        self.neurons  = self.input_neurons + self.output_neurons

        # -------- başlangıç kenarları --------
        self.edges: list[Edge] = []
        for i in self.input_neurons:
            targets = self.output_neurons
            prob    = 1.0 
            for o in targets:
                if random.random() < prob:
                    self.edges.append((i, o, random.uniform(-1, 1), True))

        self._build_layers()     # topolojik sıra hazırla
    # ------------------------------------------------------------------ #

    # --------------------- TOPOLOJİK SIRA HESABI ---------------------- #
    def _build_layers(self) -> None:
        """
        «Etkin» kenarları kullanarak katmanları çıkarır.
        Döngü kalırsa tüm kalan nöronlar son katmana atılır.
        """
        layers: list[list[int]] = [self.input_neurons.copy()]
        remaining = set(self.hidden_neurons + self.output_neurons)

        while remaining:
            current = []
            for nid in list(remaining):
                incoming = [f for f, t, _, en in self.edges if en and t == nid]
                if all(f in sum(layers, []) for f in incoming):
                    current.append(nid)
                    remaining.remove(nid)

            if not current:                     # geri besleme => döngü
                current = list(remaining)
                remaining.clear()

            layers.append(current)

        self.layers = layers  # ör.: [[in], [hid1], [hid2], [out]]

    # --------------------------- DÜŞÜNME ------------------------------ #
        # --------------------------- DÜŞÜNME ------------------------------ #
    def think(self, inputs: list[float], return_hidden=False) -> tuple[float, float]:
        """
        Girdi listesini (-1…1) aralığında iki çıktı (dv, dθ) olarak döndürür.
        """
        assert len(inputs) == self.input_size, "Input boyutu uyumsuz."

        activ: dict[int, float] = {DynamicBrain.BIAS_ID: 1.0}

        # bias dışındaki input’lar
        for i, nid in enumerate(self.input_neurons[1:]):
            activ[nid] = inputs[i]

        # katman katman hesapla
        for layer in self.layers[1:]:
            for nid in layer:
                s = 0.0
                for f, t, w, en in self.edges:
                    if en and t == nid:
                        s += w * activ.get(f, 0.0)
                activ[nid] = math.tanh(s)

        # iki çıktıyı tuple olarak döndür
        dv = activ.get(self.output_neurons[0], 0.0)
        dtheta = activ.get(self.output_neurons[1], 0.0)
        if return_hidden:
            # örnek olarak dv/dθ’yi içeren 2-D vektör
            hidden_vec = torch.tensor([dv, dtheta], dtype=torch.float32)
            return dv, dtheta, hidden_vec
        return dv, dtheta


    # --------------------------- MUTASYON ----------------------------- #
    def mutate(self) -> None:
        # 1) Ağırlıkları salla (%10)
        for idx, (f, t, w, en) in enumerate(self.edges):
            if random.random() < 0.10:
                self.edges[idx] = (f, t, w + np.random.normal(0, 0.1), en)

        # 2) Kenarı etkin/pasif yap (%05)
        #for idx, (f, t, w, en) in enumerate(self.edges):
        #    if random.random() < 0.05:
        #        self.edges[idx] = (f, t, w, not en)

        # 3) Yeni kenar (%05)
        if random.random() < 0.1:
            fr = random.choice(self.input_neurons + self.hidden_neurons)
            to = random.choice(self.hidden_neurons + self.output_neurons)
            if fr != to and not self.connection_exists(fr, to):
                self.edges.append((fr, to, random.uniform(-1, 1), True))

        # 4) Yeni nöron (%02)
        if random.random() < 0.08 and self.edges:
            idx = random.randrange(len(self.edges))
            fr, to, w, en = self.edges.pop(idx)              # kenarı yar
            new_id = 1000 + len(self.hidden_neurons)
            self.hidden_neurons.append(new_id)
            self.neurons.append(new_id)
            self.edges.append((fr,  new_id, 1.0, True))
            self.edges.append((new_id, to,  w,   True))

        self._build_layers()   # topoloji güncellendi

    # --------- YAŞARKEN UFACIK EVRİM (mutate_small) ------------------- #
    def mutate_small(self):
        """
        Online evrim için: sadece ağırlığı küçük oynatır.
        """
        for idx, (f, t, w, en) in enumerate(self.edges):
            if random.random() < self.mutation_rate * 0.1:
                self.edges[idx] = (f, t, w + np.random.normal(0, 0.02), en)

    # --------------------------- YARDIMCI ----------------------------- #
    def connection_exists(self, f, t) -> bool:
        return any(fr == f and to == t for fr, to, _, _ in self.edges)

    # ----------------------------- KLON ------------------------------- #
    def clone(self) -> "DynamicBrain":
        child = DynamicBrain(self.input_size, self.output_size,
                             self.mutation_rate)
        child.hidden_neurons = self.hidden_neurons.copy()
        child.neurons        = self.neurons.copy()
        child.edges          = [(f, t, w, en) for f, t, w, en in self.edges]
        child.mutate()                 # çocukta mutasyon uygula
        child._build_layers()
        return child

    # --------------------------- ÇİZ (pygame) -------------------------- #
    def draw(self, surface, panel_x, panel_w, font):
        """
        Basit görselleştirme. Bias nöronu INPUT sütununda en üstte çizilir.
        """
        if surface is None:
            return

        start_y        = 300
        neuron_spacing = 18
        layer_spacing  = 90

        # Bias + diğer inputlar
        total_inputs = len(self.input_neurons)
        input_pos  = [(panel_x + 40, start_y + i*neuron_spacing)
                      for i in range(total_inputs)]
        hidden_pos = [(panel_x + 40 + layer_spacing,
                       start_y + i*neuron_spacing)
                      for i in range(len(self.hidden_neurons))]
        output_pos = [(panel_x + 40 + 2*layer_spacing,
                       start_y + i*neuron_spacing)
                      for i in range(len(self.output_neurons))]

        neuron_pos = {}
        for idx, nid in enumerate(self.input_neurons):
            neuron_pos[nid] = input_pos[idx]
        for idx, nid in enumerate(self.hidden_neurons):
            neuron_pos[nid] = hidden_pos[idx]
        for idx, nid in enumerate(self.output_neurons):
            neuron_pos[nid] = output_pos[idx]

        # Bağlantılar
        for fr, to, w, en in self.edges:
            if fr in neuron_pos and to in neuron_pos and en:
                s = neuron_pos[fr]
                e = neuron_pos[to]
                color = BLUE if w > 0 else RED
                width = max(1, int(abs(w)*3))
                pygame.draw.line(surface, color, s, e, width)

        # Nöronlar
        for nid, pos in neuron_pos.items():
            if nid == DynamicBrain.BIAS_ID:
                size = 5
                pygame.draw.circle(surface, (255, 165, 0), (int(pos[0]), int(pos[1])), size)  # turuncu bias
            elif nid in self.input_neurons:
                pygame.draw.circle(surface, BLACK, (int(pos[0]), int(pos[1])), 4)
            elif nid in self.hidden_neurons:
                pygame.draw.circle(surface, BLACK, (int(pos[0]), int(pos[1])), 6)
            else:
                pygame.draw.circle(surface, BLACK, (int(pos[0]), int(pos[1])), 8)

        # Başlıklar
        font_small = pygame.font.SysFont(None, 18)
        surface.blit(font_small.render("INPUT",  True, BLACK),
                     (panel_x + 25, start_y - 25))
        surface.blit(font_small.render("HIDDEN", True, BLACK),
                     (panel_x + 25 + layer_spacing, start_y - 25))
        surface.blit(font_small.render("OUTPUT", True, BLACK),
                     (panel_x + 25 + 2*layer_spacing, start_y - 25))

    # ------------------------ SERİYALİZASYON -------------------------- #
    def to_dict(self):
        return {
            "input_neurons":  self.input_neurons,
            "hidden_neurons": self.hidden_neurons,
            "output_neurons": self.output_neurons,
            "edges":          self.edges,
            "mutation_rate":  self.mutation_rate
        }

    @classmethod
    def from_dict(cls, data):
        brain = cls(len(data["input_neurons"]) - 1,       # bias hariç
                    len(data["output_neurons"]),
                    data["mutation_rate"])
        brain.input_neurons  = data["input_neurons"]
        brain.hidden_neurons = data["hidden_neurons"]
        brain.output_neurons = data["output_neurons"]
        brain.neurons        = (brain.input_neurons +
                                brain.hidden_neurons +
                                brain.output_neurons)
        brain.edges          = [tuple(e) for e in data["edges"]]
        brain._build_layers()
        return brain
