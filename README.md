# Rags2Riches: Computational Garment Reuse
![Rags2Riches](images/teaser.png)
In this repository, we share the core code for the prototype implementation of the research paper:

> *“Rags2Riches: Computational Garment Reuse”*  
> Anran Qi, Nico Pietroni, Maria Korosteleva, Olga Sorkine-Hornung and Adrien Bousseau, 
> SIGGRAPH Conference Papers ’25

Our algorithm 

- takes as input two garment designs along with their corresponding sewing patterns and
determines how to cut one of them to match the other by following garment
reuse principles. 

- Specifically, our algorithm favors the reuse of seams and
hems present in the existing garment, thereby preserving the embedded
value of these structural components and simplifying the fabrication of the
new garment.

More details can be found at: [[project page]](https://) | [[paper]](https://anranqi.github.io/img/rags2riches_author_compress.pdf)  

# Setup:
 ```bash
git clone git@github.com:graphdeco-inria/rags2riches.git
cd rags2riches
 ```
- install pygco following instruction https://github.com/Borda/pyGCO
- pip install time, numpy, matplotlib, ortools, skimage, igl, typing
- run the demo by
 ```bash
python demo_pants2bag.py
 ```

# Credits
- The pattern class was adapted from Maria's https://github.com/maria-korosteleva/Garment-Pattern-Generator/tree/master/packages/pattern
- svgpathtools was slightly modified from https://pypi.org/project/svgpathtools/

# Acknowledgments
We thank the anonymous reviewers for their valuable feedback. We
also thank Glwadys Milong for making our physical prototype. This
work was supported in part by the ERC Consolidator Grant No.
101003104 (MYCLOTH) and the PHC FASIC 2025 program, project
53593QK.
# Contact&License
Please let us know (annranqi@gmail.com) if you have any question regarding the algorithms/paper or you find any bugs in the code :) 
This work is licensed under a [Creative Commons Attribution-NonCommercial 4.0 International License](http://creativecommons.org/licenses/by-nc/4.0/). For any commercial uses or derivatives, please contact us