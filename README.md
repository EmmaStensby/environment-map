# Evolving environments and modular robots with MAP-Elites

This repository contains the source code for our paper *Open-ended search for environments and adapted agents using MAP-Elites*.

**Abstract.** Creatures in the real world constantly encounter new and diverse challenges they have never seen before. They will often need to adapt to some of these tasks and solve them in order to survive. This almost endless world of novel challenges is not as common in virtual environments, where artificially evolving agents often have a limited set of tasks to solve. An exception to this is the field of open-endedness where the goal is to create unbounded exploration of interesting artefacts. We want to move one step closer to creating simulated environments similar to the diverse real world, where agents can both find solvable tasks, and adapt to them. Through the use of MAP-Elites we create a structured repertoire, a map, of terrains and virtual creatures that locomote through them. By using novelty as a dimension in the grid, the map can continuously develop to encourage exploration of new environments. The agents must adapt to the environments found, but can also search for environments within each cell of the grid to find the one that best fits their set of skills. Our approach combines the structure of MAP-Elites, which can allow the virtual creatures to use adjacent cells as stepping stones to solve increasingly difficult environments, with open-ended innovation. This leads to a search that is unbounded, but still has a clear structure. We find that while handcrafted bounded dimensions for the map lead to quicker exploration of a large set of environments, both the bounded and unbounded approach manage to solve a diverse set of terrains.


The article can be found at:

https://link.springer.com/chapter/10.1007/978-3-031-02462-7_41

Cite as:



<pre>
Norstein, E.S., Ellefsen, K.O., Glette, K. (2022). Open-Ended Search for Environments and Adapted Agents Using MAP-Elites. In: Jiménez Laredo, J.L., Hidalgo, J.I., Babaagba, K.O. (eds) Applications of Evolutionary Computation. EvoApplications 2022. Lecture Notes in Computer Science, vol 13224. Springer, Cham. https://doi.org/10.1007/978-3-031-02462-7_41
</pre>



```
@InProceedings{10.1007/978-3-031-02462-7_41,
author="Norstein, Emma Stensby
and Ellefsen, Kai Olav
and Glette, Kyrre",
editor="Jim{\'e}nez Laredo, Juan Luis
and Hidalgo, J. Ignacio
and Babaagba, Kehinde Oluwatoyin",
title="Open-Ended Search for Environments and Adapted Agents Using MAP-Elites",
booktitle="Applications of Evolutionary Computation",
year="2022",
publisher="Springer International Publishing",
address="Cham",
pages="651--666",
abstract="Creatures in the real world constantly encounter new and diverse challenges they have never seen before. They will often need to adapt to some of these tasks and solve them in order to survive. This almost endless world of novel challenges is not as common in virtual environments, where artificially evolving agents often have a limited set of tasks to solve. An exception to this is the field of open-endedness where the goal is to create unbounded exploration of interesting artefacts. We want to move one step closer to creating simulated environments similar to the diverse real world, where agents can both find solvable tasks, and adapt to them. Through the use of MAP-Elites we create a structured repertoire, a map, of terrains and virtual creatures that locomote through them. By using novelty as a dimension in the grid, the map can continuously develop to encourage exploration of new environments. The agents must adapt to the environments found, but can also search for environments within each cell of the grid to find the one that best fits their set of skills. Our approach combines the structure of MAP-Elites, which can allow the virtual creatures to use adjacent cells as stepping stones to solve increasingly difficult environments, with open-ended innovation. This leads to a search that is unbounded, but still has a clear structure. We find that while handcrafted bounded dimensions for the map lead to quicker exploration of a large set of environments, both the bounded and unbounded approach manage to solve a diverse set of terrains.",
isbn="978-3-031-02462-7"
}

```
