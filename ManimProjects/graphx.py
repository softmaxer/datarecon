from manimlib.imports import *
import numpy as np


class Intro(Scene):
    def construct(self):
        first_line = TextMobject("Fast Fourier Transforms")
        second_line = TextMobject("powered by manim")
        second_line.next_to(first_line, DOWN)
        self.play(Write(first_line))
        self.wait(1)
        self.play(Transform(first_line, second_line))
        self.wait(1)
        self.play(FadeOut(second_line))

class graphx(GraphScene):
    CONFIG={
            "x_min": -4,
            "x_max": 4,
            "y_min": -4,
            "y_max": 4,
            "axes_color": BLUE,
            "graph_origin": 0 * DOWN + 0* LEFT,
            "x_axis_label": "",
            "y_axis_label": ""
            }

    def construct(self):
        # Text
        noisy = TextMobject("Noisy signal.")
        noisy.set_color(RED)
        noisy.to_edge(DOWN)
        clean = TextMobject("Clean signal.")
        clean.set_color(BLUE)
        clean.to_edge(DOWN)
        components = TextMobject("Pure sine components.")
        components.set_color(YELLOW)
        components.to_edge(UP)
        # Graphs
        self.setup_axes(animate=True)
        graph = self.get_graph(lambda x: np.sin(2*np.pi*50*x) + np.sin(2*np.pi*120*x) + np.random.random(), x_min=-np.pi, x_max=np.pi)
        graph.set_color(RED)
        graph2 = self.get_graph(lambda x: np.sin(2*np.pi*50*x) + np.sin(2*np.pi*120*x), x_min=-np.pi, x_max=np.pi)
        graph2.set_color(BLUE)
        pure1 = self.get_graph(lambda x: np.sin(2*np.pi*50*x), x_min=-np.pi, x_max=np.pi)
        pure1.set_color(YELLOW)
        pure2 = self.get_graph(lambda x: np.sin(2*np.pi*120*x), x_min=-np.pi, x_max=np.pi)
        pure2.set_color(PURPLE)
        # Animations
        self.play(ShowCreation(graph), Write(noisy))
        self.wait(1)
        self.play(Transform(graph, graph2))
        self.play(Transform(noisy, clean))
        self.wait(1)
        self.play(Transform(graph2, pure1)) 
        clean.set_color(BLACK)
        self.play(Transform(clean, components))
        self.wait(1)
        self.play(Transform(pure1, pure2))
        components.match_color(pure2)
        self.play(Write(components))
        self.wait(2)
        self.play(FadeOut(pure2))
