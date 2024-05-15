METODY STROJOVÉHO UČENÍ HLUBOKÝCH NEURONOVÝCH SÍTÍ S OMEZENÝMI DATASETY 

Priečinok obsahuje zdrojové kódy vypracované na experimenty diplomovej práce a výsledky z nich uložené v json formáte.
Natrénované modely nie sú súčasťou z dôvodu vysokej veľkosti, ale sú nahrané na dostupnom odkaze nižšie.
* https://drive.google.com/drive/folders/1aq2wplMjM2-MYvT5gB9ULy3NGBSO5cbg?usp=sharing

Realizácia experimentov :

* Spyder version: 5.4.3  (conda)
* Python version: 3.11.8 64-bit
* Qt version: 5.15.2
* PyQt5 version: 5.15.10
* Operating System: Linux 6.5.0-27-generic
* Nvidia Titan Xp 12 GB GDDR5


Označenia :
* CB (Car/Bike) - Dataset áut a motoriek 
* LC (Lung cancer) - Dataset obrazov z histológie

Priečinok "scripts":

0_main_full_dataset
-Experimentálny skript na nastavenie parametrov s plným datasetom.

1_decreasing_dataset
-Skript na znižovanie počtu obrazov na triedu v trénovacej množine. Učenie s odlišnými architektúrami a epochami.

2_transfer_learning
-Skript na znižovanie počtu obrazov na triedu v trénovacej množine. Učenie s použitím natrénovaných váh (verzia IMAGENET1K_V1)

3_augmentation_online
-Skript na znižovanie počtu obrazov na triedu v trénovacej množine. Učenie s dátovovu augmentáciou online.

4_augmentation_online
-Skript na znižovanie počtu obrazov na triedu v trénovacej množine. Učenie s dátovovu augmentáciou offline s použitím vytvorenej funkcie augmentation_function.

5_NST_experiment
-Skript na znižovanie počtu obrazov na triedu v trénovacej množine. Učenie s dátovou augmentáciou pomocou NST.
-Pred ním vykonaná augmentácia pre jednotlivé triedy pomocou skriptov, ktoré pracujú s hlavným skriptom NST_main : NST_bike, 
							     							   NST_car,
							     							   NST_lunc_aca, 
							     						           NST_lung_scc

-Skript NST_main je čiastočne inšpirovaný tutoriálom použitia siete : https://github.com/pytorch/tutorials/blob/main/advanced_source/neural_style_tutorial.py
-Výsledné obrazy dostupné v zložke NST_images na : https://drive.google.com/drive/folders/1aq2wplMjM2-MYvT5gB9ULy3NGBSO5cbg?usp=sharing


Priečinok "plotting" :

- Výsledky sú uložené v priečinku "plotting" pre každý dataset zvlášť v json formáte, ktoré sú pomenované na základe výstupu zo skriptov.
- Obsahuje 2 priečinky (Car Bike plotting, Lung cancer plotting)
- Priečinky obsahujú súbory plotting_CB/plotting_LC, ktoré vykresľujú všetky grafy na základe výsledkov v json formáte.




