### Appendix. Remote Sensing and Machine Learning-Based Tungsten Prospectivity Mapping in Primorsky Krai, Russian Far East
**Sergei Shevyrev, Irina Tarasenko, and Natalia Boriskina**

*FEGI FEB RAS, Vladivostok, Russia*

### Processing of topographically corrected Landsat 8 OLI bands to compute mineral components
The algorithm uses directed principal components method. It selects all possible two band ratios using Python for every mineral 
("quartz","alunite", "illite","chlorite","goethite","epidote") and noise signal of “vegetation” value based on a rule: mineral and vegetation band ratios  must be greater that 1 for vegetatoin reflectances, when for mineral they can be greater or lower than 1.
Possibly "mineral" and "vegetation" components are saved into filenames stored in MINERAL_RATIOS and VEGETATION_RATIOS.

Reflectance values for "minerals" and "vegetations" are taken from [1,2].

***References***
<ol>
<li>R.N. Clark, G.A. Swayze, A. Gallagher, T.V.V. King, W.M. Calvin The US Geological Survey Digital Spectral Library, Version 1:0.2 to 3.0 mm US Geological Survey Open File Report (1993), pp. 93-592 </li>
<li>Hubbard, B. E., Crowley, J. K., Zimbelman D.R., 2003 Comparative alteration mineral mapping using visible to shortwave infrared (0.4–2.4μm) Hyperion, ALI and ASTER Imagery. IEEE Trans. Geosci. Remote Sens., 41 1401 –1410 (2003). doi: 10.1109/TGRS.2003.812906IGRSD20196-2892.</li>
</ol>
