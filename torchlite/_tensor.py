from __future__ import annotations

import numpy as np
import typing as t
from typing import Union
from graphviz import Digraph
from functools import partial


class Tensor:
    def __init__(self,value: np.ndarray[t.Any, np.dtype[t.Any]] , label = str()) -> None:
        
        self.children: tuple[Tensor] = ()
        self.operator = str()
        self.label = label

        # if isinstance(value, (np.ndarray,list,tuple)):
        #     self.value = np.array(value)

        # if isinstance(value,Tensor):
        #     self.value=value.value    

        if isinstance(value,Tensor):
            self.value=value.value  
        else:
            self.value = np.array(value)
        

        self.grad = 0.0
        self.grad_fn=lambda:None

    def __repr__(self):

        if isinstance(self.value, float):
            value = f'({f"{self.value:.4f}".rstrip("0")}'

        elif isinstance(self.value, np.ndarray):
            value = np.array_repr(self.value, precision=4).strip("array")

        return f"Tensor({value})"
    
    def __add__(self,other:Tensor | np.ndarray[t.Any, np.dtype[t.Any]]) -> Tensor:

        other = other if isinstance(other,Tensor) else Tensor(other)
        result = Tensor(self.value + other.value)
        result.children = (self,other)
        result.operator = '+'
        
        def addition_grad_fn() -> None:
            self.grad += result.grad
            other.grad += result.grad

        result.grad_fn = addition_grad_fn
        return result
    
    def __radd__(self,other) -> Tensor:
        return self + Tensor(other)
    
    def __sub__(self,other) -> Tensor:
        return self + (-Tensor(other))
    
    def __rsub__(self, other): # other - self
        return Tensor(other) + (-self)
    
    def __mul__(self,other:Tensor | np.ndarray[t.Any, np.dtype[t.Any]]) -> Tensor:

        other = other if isinstance(other, Tensor) else Tensor(other)
        result = Tensor(self.value * other.value)
        result.children = (self,other)
        result.operator = '*'

        def multiplication_grad_fn() -> None:
            self.grad += other.value * result.grad
            other.grad += self.value * result.grad

        result.grad_fn = multiplication_grad_fn
        return result  
    def __rmul__(self,other) -> Tensor:
        return self * Tensor(other)
        
    def __pow__(self,power:int | float):

        assert isinstance(power,(int,float)), 'Value of power paramenter should be of type int or float'
        result = Tensor(self.value ** power)
        result.children = (self,)
        result.operator = '**'

        def power_grad_fn() -> None:
            self.grad = (power * self.value ** (power - 1)) * result.grad

        result.grad_fn = power_grad_fn
        return result
    
    def __neg__(self) -> Tensor:
        return self * Tensor([-1])
    
    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1
    
   
    

      
    def __array_ufunc__(
        self, ufunc: np.ufunc, method: str, *inputs: t.Any, **kwargs: t.Any
    ) -> Tensor | np.ndarray | int | float | complex | None:
        other = [
            input.data if isinstance(input, Tensor) else input
            for input in inputs
        ]
        result = getattr(ufunc, method)(*other, **kwargs)
        if isinstance(result, np.ndarray):
            return Tensor(result)
        return result 
            
    def _generate_nodes_and_edges(self):

        nodes, edges = set(), set()

        def populate_nodes_and_edges(self: Tensor) -> None:
            if self not in nodes:
                nodes.add(self)
                for node in self.children:
                    edges.add((node, self))
                    populate_nodes_and_edges(node)

        populate_nodes_and_edges(self)
        return nodes, edges
    
    def visualize(self):

        graph = Digraph(name="Computation Graph", graph_attr={"rankdir": "LR"})
        nodes, edges = self._generate_nodes_and_edges()

        for node in nodes:
            tensor = str(id(node))
            graph.node(
                name=tensor,
                label=(
                    #f"node = {node.label}\n"
                    f"data = {np.array_repr(node.value,precision=4)}\n"
                    f"grad = {np.array_repr(np.array(node.grad),precision=4)}\n"
                ),
                shape="circle",
                style="filled",
                width="0.02",
            )
            if node.operator:
                operator = tensor + node.operator
                graph.node(name=operator, label=node.operator)
                graph.edge(operator, tensor)

        for node_l, node_r in edges:
            graph.edge(str(id(node_l)), str(id(node_r)) + node_r.operator)

        graph.view(cleanup=True)
    
    def backward(self) -> None:
        graph: list[Tensor] = []
        visited: set[Tensor] = set()

        def create_graph(node: Tensor) -> None:
            if node not in visited:
                visited.add(node)
                for _node in node.children:
                    create_graph(_node)
                graph.append(node)

        create_graph(self)
        self.grad = 1.0
        for node in reversed(graph):
            node.grad_fn()

        self.visualize()
        
    
# a=Tensor((3.0,),'A')
# b=Tensor(np.array([2,3]),'B')
# c=a*b
# c.label='C'
# print(c.backward())

# d=Tensor((3.0,),'D')
# e=Tensor(np.array([2,3]),'E')
# f=d+e
# f.label='F'
# print(f.backward())

# x=Tensor((3.0,),'X')
# w=Tensor((3.0,6,),'W')
# b=Tensor([2],'b')


# #i=(g**h)+a
# y=3/x
# y.label='Y'
# print(y.backward())
