# Copyright © 2018 Battelle Memorial Institute
# All rights reserved.

import numpy as np
import pandas as pd
from itertools import islice, chain, repeat

import networkx as nx

import matplotlib.pyplot as plt

import hypernetx as hnx

__all__ = ["LesMis", "lesmis_hypergraph_from_df", "book_tour"]


class LesMis(object):
    def __init__(self):
        self.volumes = pd.DataFrame.from_dict(volume_names, orient="index")

        accents = {"\`e": "è", "\\`e": "è", "'e": "é", "\\c{c}": "ç", "\^o": "ô"}
        for k, v in accents.items():
            self.names = names.replace(k, v)

        self.df_names = pd.DataFrame(
            [parse_name_row(row) for row in self.names.split("\n")],
            columns=["Symbol", "FullName", "Description"],
        )

        self.df_scenes = pd.DataFrame(
            list(get_scene_data()),
            columns=["Volume", "Book", "Chapter", "Scene", "Step", "Characters"],
        )

        self.book_tour_data = self.df_scenes.groupby(["Volume", "Book"]).apply(
            lesmis_hypergraph_from_df, by="Chapter"
        )

    @property
    def dnames(self):
        return self.df_names.set_index("Symbol")


def lesmis_hypergraph_from_df(df, by="Chapter", on="Characters"):
    cols = df.columns.tolist()

    return hnx.Hypergraph(
        {
            ".".join(map(str, t)): set(dft)
            for t, dft in df.groupby(cols[: cols.index(by) + 1])[on]
        }
    )


def book_tour(df, xlabel="Book", ylabel="Volume", s=3.5):
    """
    Constructs a visualization of hypergraphs stored in an indexed
    dict like object. See `Tutorial 4 - LesMis Visualizations-BookTour
    <https://github.com/pnnl/HyperNetX/blob/master/tutorials/Tutorial%204%20-%20LesMis%20Visualizations-BookTour.ipynb>`_

    Parameters
    ----------
    df : TYPE
        Description
    xlabel : str, optional
        Description
    ylabel : str, optional
        Description
    s : float, optional
        Description

    Yields
    ------
    TYPE
        Description
    """
    df_book_tour = df
    # df_book_tour = book_tour_data()

    nrows, ncols = df_book_tour.index.max()

    plt.figure(figsize=(s * ncols, s * nrows))
    for (v, b), G in df_book_tour.items():
        ax = plt.subplot(nrows, ncols, (v - 1) * ncols + b, label=f"{v}.{b}")

        ax.set_xlabel(f"{xlabel} {b}")
        if b == 1:
            ax.set_ylabel(f"{ylabel} {v}")
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])

        yield (v, b), G, ax


# Helper Functions:
def parse_name_row(row):
    abbrev, descr = row.split(" ", 1)
    fullname, descr = descr.split(",")
    return abbrev, fullname, descr


def get_scene_data():
    t = 0
    for row in interactions.split("\n"):
        numbers, characters = islice(chain(row.split(":"), repeat(None)), 2)

        volume, book, chapter = numbers.split(".")

        if characters:
            for i, si in enumerate(characters.split(";")):
                for c in si.split(","):
                    yield int(volume), int(book), int(chapter), i, t, c
                t += 1


# LesMis Data:
names = """AZ Anzelma, daughter of TH and TM
BA Bahorel, `Friends of the ABC' cutup
BB Babet, tooth-pulling bandit of Paris
BJ Brujon, notorious criminal
BL Blacheville, Parisian student from Montauban
BM Monsieur Bamatabois, idler of M-- sur M--
BO Bossuet (Lesgle), `Friends of the ABC' klutz
BR Brevet, convict in the galleys with JV
BS Bruneseau, explorer and mapper of the sewers of Paris
BT Baroness of T--, friend of GI
BU Madame Burgon, new landlady at Gorbeau House
BZ Boulatruelle, former convict and road mender in Montfermeil
CC Cochepaille, convict in the galleys with JV
CH Champmathieu, accused thief mistaken for JV
CL Countess de L\^o, distant relative of MY
CM Combeferre, `Friends of the ABC' guide
CN Chenildieu, convict in the galleys with JV
CO Cosette, daughter of FN and FT
CR Courfeyrac, `Friends of the ABC' center
CV Cravatte, mountain bandit
DA Dahlia, lover of LI
EN Enjolras, `Friends of the ABC' chief
EP Eponine, daughter of TH and TM
FA Fameuil, Parisian student from Limoges
FE Feuilly, `Friends of the ABC' political idealist
FF Fauchelevent, aged notary of M-- sur M--
FN Fantine, lover of FT
FT F\'elix Tholomy\`es, Parisian student from Toulouse
FV Favourite, lover of BL
GA Gavroche, young urchin living at Gorbeau House
GE G\'eborand, retired merchant of D--
GG G--, former member of National Convention
GI Monsieur Luke Esprit Gillenormand, grand bourgeois
GP George Pontmercy, father of MA and son-in-law of GI
GR Gribier, new gravedigger at cemetery
GT Grantaire, `Friends of the ABC' skeptic
GU Gueulemer, Herculean bandit of Paris
HL Madame Hucheloup, keeper of Corinth Inn
IS Isabeau, baker
JA Javert, police officer of M-- sur M--
JD Jondrette, father of GA
JL Jacquin Labarre, innkeeper of La Croix de Calbas
JO Joly, `Friends of the ABC' medic
JP Jean Prouvaire, `Friends of the ABC' poet
JU Judge of Douai, judge at the court trying CH
JV Jean Valjean, thief of bread
LI Listolier, Parisian student from Cahors
LL Old woman 2, landlady of JV in Paris at Gorbeau House
LP Louis Philippe, Orleans King of France
MA Marius, grandson of GI
MB Mademoiselle Baptistine, sister of MY
MC Marquis de Champtercier, ultra-royalist miser
ME Madame Magloire, housekeeper to MY
MG Madamoiselle Gillenormand, unmarried daughter of GI
MI Mother Innocent, prioress of Convent of Petite Rue Picpus
MM Monsieur Mabeuf, prefect of church
MN Magnon, servant of GI
MO Montparnasse, genteel bandit of Paris
MP Madame Pontmercy, younger daughter of GI
MR Madame de R--, Marquise de R--
MT Marguerite, old lady who teaches FN to live poor
MV Madamoiselle Vaubois, friend of MG
MY Monsieur Charles Fran\c{c}ois Bienvenu Myriel, Bishop of D--
NP Napoleon, Emperor of France
PG Petit Gervais, a small boy in D--
PL Mother Plutarch, maid of MM
PO Old woman 1, portress of JV in M-- sur M--
QU Claquesous, night-like bandit of Paris
SC Monsieur Scaufflaire, keeper of horses and chaises in M-- sur M--
SN Count ***, `philosophic' senator
SP Sister Perp\'etue, stout nun at infirmary in M-- sur M--
SS Sister Simplice, saintly nun at infirmary in M-- sur M--
TG Lieutenant Theodule Gillenormand, soldier and grandnephew of GI
TH Th\'enardier, sergeant of Waterloo and keeper of a chophouse
TM Madame Th\'enardier, wife of TH
TS Toussaint, servant of JV at Rue Plumet
VI Madame Victurnien, snoop in M-- sur M--
XA Child 1, son of TH sold to MN
XB Child 2, son of TH sold to MN
ZE Zephine, lover of FA"""

volume_names = {
    1: {"title": "Fantine"},
    2: {"title": "Cosette"},
    3: {"title": "Marius"},
    4: {"title": "St. Denis"},
    5: {"title": "Jean Valjean"},
}

interactions = """1.1.1:MY,NP;MY,MB
        1.1.2:MY,ME;ME,MB
        1.1.3:MY
        1.1.4:MY,ME;MY,CL;MY,GE;MY,MC;MY,MB
        1.1.5:MY,MB,ME
        1.1.6:ME,MY
        1.1.7:MY,CV;MY,MB,ME
        1.1.8:SN,MY
        1.1.9:MB
        1.1.10:MY,GG
        1.1.11:MY
        1.1.12:MY
        1.1.13:MY
        1.1.14:MY,SN
        1.2.1:JL,JV;JV,MT;MR,JV
        1.2.2:ME,MB,MY
        1.2.3:ME,MB,MY,JV
        1.2.4:MY,JV,MB;MY,JV,MB,ME
        1.2.5:MY,ME,JV
        1.2.6:JV,IS
        1.2.7:JV
        1.2.8
        1.2.9:JV
        1.2.10:JV
        1.2.11:JV
        1.2.12:MY,ME;MY,JV
        1.2.13:PG,JV
        1.3.1
        1.3.2:FT,LI,FA,BL
        1.3.3:FT,LI,FA,BL,FV,DA,ZE,FN
        1.3.4:FT,LI,FA,BL,FV,DA,ZE,FN
        1.3.5
        1.3.6:BL,FV;FV,DA
        1.3.7:FT
        1.3.8:FT,LI,FA,BL,FV,DA,ZE,FN
        1.3.9:FV,DA,ZE,FN
        1.4.1:TM,FN;TH,TM,FN
        1.4.2
        1.4.3:CO;TH;TM
        1.5.1:JV
        1.5.2:JV
        1.5.3:JV
        1.5.4:MY
        1.5.5:JA
        1.5.6:FF,JV,JA
        1.5.7:FF
        1.5.8:VI;FN
        1.5.9:VI;MT,FN
        1.5.10:MT,FN
        1.5.11
        1.5.12:BM,FN,JA
        1.5.13:FN,JA;FN,JA,JV;JA,JV;JV,FN
        1.6.1:JV,FN
        1.6.2:JV,JA
        1.7.1:SP,SS;JV,SS;JV,FN
        1.7.2:JV,SC
        1.7.3:JV
        1.7.4:JV,PO
        1.7.5:JV
        1.7.6:SS,FN
        1.7.7:JV
        1.7.8:JV
        1.7.9:JV,JU,CH,BM
        1.7.10:JU,CH,BR,CN,CC,JV,BM
        1.7.11:JV,BR,CN,CC,JU,CH
        1.8.1:SS,JV;JV,FN
        1.8.2:JV,FN
        1.8.3:JV,FN,JA
        1.8.4:JV,FN;JV,JA;JA,FN,JV;JA,JV
        1.8.5:JV,PO;FN,SP,SS;JV,SS;PO,JA;JA,SS
        2.1.1
        2.1.2
        2.1.3
        2.1.4:NP
        2.1.5
        2.1.6
        2.1.7:NP
        2.1.8:NP
        2.1.9:NP
        2.1.10:NP
        2.1.11
        2.1.12
        2.1.13
        2.1.14
        2.1.15
        2.1.16
        2.1.17
        2.1.18
        2.1.19:TH,GP
        2.2.1:JV
        2.2.2:TH,BZ
        2.2.3
        2.3.1:CO
        2.3.2:TH,TM
        2.3.3:TM,CO
        2.3.4:TM,CO
        2.3.5:CO
        2.3.6:JV,CO
        2.3.7:CO,JV
        2.3.8:TM,JV;CO,TM;JV,TM;JV,TH,TM;TH,TM;EP,AZ;EP,TM;TM,CO;TH,JV
        2.3.9:TM,JV;TH,JV
        2.3.10:TH,JV
        2.3.11
        2.4.1:CO,JV
        2.4.2:CO,JV
        2.4.3:CO,JV;JV,LL
        2.4.4:LL,JV,CO
        2.4.5:JA,JV;JV,CO;JV,LL
        2.5.1:CO,JV
        2.5.2:CO,JV
        2.5.3:CO,JV
        2.5.4:CO,JV
        2.5.5:CO,JV
        2.5.6:CO,JV
        2.5.7:CO,JV
        2.5.8:JV,FF
        2.5.9:JV,FF
        2.5.10:JA,TH;JA,LL;JA,JV
        2.6.1
        2.6.2
        2.6.3
        2.6.4
        2.6.5
        2.6.6
        2.6.7:MI
        2.6.8
        2.7.1
        2.7.2
        2.7.3
        2.7.4
        2.7.5
        2.7.6
        2.7.7
        2.7.8
        2.8.1:JV,FF
        2.8.2:FF,MI
        2.8.3:FF,MI
        2.8.4:FF,JV;JV,CO
        2.8.5:GR,FF
        2.8.6:JV
        2.8.7:FF,GR;FF,JV
        2.8.8:FF,MI,JV
        2.8.9:FF,JV
        3.1.1
        3.1.2
        3.1.3
        3.1.4
        3.1.5
        3.1.6
        3.1.7
        3.1.8:JD,BU;GA,BU
        3.2.1:GI
        3.2.2:GI
        3.2.3:GI
        3.2.4:GI
        3.2.5:GI
        3.2.6:GI,MN
        3.2.7:GI
        3.2.8:MG,MP;MG,MV;MG,TG;MG,GI,MA
        3.3.1:BT,GI;MG,GI,MA
        3.3.2:GP,MP;MG,MA
        3.3.3:BT,MA
        3.3.4:GI,MA;MA,GP
        3.3.5:MA,MM;GI,MA,MG
        3.3.6:MA
        3.3.7:TG,MG;TG,MA
        3.3.8:GI,MG;GI,MA;GI,MG
        3.4.1:EN;CM;JP;FE;CR;BA;BO;JO;GT
        3.4.2:MA,BO;MA,BO,CR
        3.4.3:CR,MA;EN,CR,MA
        3.4.4:GT,BO;JO,BA
        3.4.5:CR,EN,MA,CM
        3.4.6:CR,MA
        3.5.1:MA
        3.5.2:MA
        3.5.3:MA
        3.5.4:MM,PL
        3.5.5:MM
        3.5.6:GI,MG;GI,TG
        3.6.1:MA,CO;MA,JV,CO
        3.6.2:MA,JV,CO
        3.6.3:MA,CO
        3.6.4:MA
        3.6.5:MA
        3.6.6:MA,CO;MA,CO,FT
        3.6.7:MA
        3.6.8:MA,CO
        3.6.9:MA
        3.7.1
        3.7.2
        3.7.3:GU;BB;QU;MO
        3.7.4
        3.8.1:MA
        3.8.2:MA
        3.8.3:MA
        3.8.4:MA,TH
        3.8.5:MA
        3.8.6:TH
        3.8.7:TH,EP;TH,TM
        3.8.8:TH,JV,CO
        3.8.9:TH,JV
        3.8.10:MA
        3.8.11:MA,EP
        3.8.12:TH,TM
        3.8.13:MA
        3.8.14:MA,JV
        3.8.15:CR,BO
        3.8.16:TH,TM,EP,AZ
        3.8.17:TH,TM
        3.8.18:TH,TM,JV
        3.8.19:TH,TM,JV
        3.8.20:TH,BB,GU,QU;TH,JV;TH,QU;TH,BB,GU,QU,JV,TM;TH,JV;TH,TM;TH,JA
        3.8.21:TH,JA,BB;TM,TH,JA;JA,BB,GU,QU,MO;JA,JV
        3.8.22:BU,GA
        4.1.1
        4.1.2
        4.1.3:LP
        4.1.4:LP
        4.1.5
        4.1.6:EN,CM,CR,GT
        4.2.1:MA
        4.2.2
        4.2.3:MM,EP
        4.2.4:MA,EP
        4.3.1:JV
        4.3.2:JV
        4.3.3
        4.3.4:JV,CO
        4.3.5:JV,CO
        4.3.6:CO,MA
        4.3.7:JV,CO
        4.3.8:JV,CO
        4.4.1:JV,CO
        4.4.2:MM,PL;MO,JV
        4.5.1:CO,TG
        4.5.2:JV,CO
        4.5.3:CO,TS
        4.5.4:MA
        4.5.5:CO
        4.5.6:CO,MA
        4.6.1:TM,MN
        4.6.2:GA,XA,XB;GA,MO;GA,XA,XB
        4.6.3:BB,BJ,GU,TH;BB,BJ,GU,TH,GA
        4.7.1
        4.7.2
        4.7.3
        4.7.4
        4.8.1:CO,MA
        4.8.2:CO,MA
        4.8.3:MA,CR;MA,CO;MA,EP
        4.8.4:EP,BB,BJ,GU,TH,QU,MO
        4.8.5
        4.8.6:MA,CO
        4.8.7:GI,MG;GI,MA
        4.9.1:JV
        4.9.2:MA,CR;MA,EP
        4.9.3:MM,PL
        4.10.1
        4.10.2
        4.10.3
        4.10.4
        4.10.5
        4.11.1:GA
        4.11.2:GA
        4.11.3:GA
        4.11.4:CM,GA,CR,BA
        4.11.5:CM,CR,BA,MM
        4.11.6:CR,EP
        4.12.1:HL
        4.12.2:BO,JO,GT,HL
        4.12.3:GA,BA,CR,BO,EN,FE,JO,GT,JP
        4.12.4:BA,CR,HL,GA,EN
        4.12.5:CR,EN
        4.12.6:EN,CM,CR,JP,FE,BO,JO,BA
        4.12.7:EN,JA;GA,JA;EN,JA
        4.12.8:QU,EN;EN,CM,JP
        4.13.1:MA
        4.13.2
        4.13.3:MA
        4.14.1:EN,CM,GA;EN,CM,CR,BO,JO,BA,GA,FE,MM
        4.14.2:MM
        4.14.3:EN,JA;CR,EN,JP;CM,JO,BA,BO,GA,MA
        4.14.4:MA
        4.14.5:MA,CR,CM,BO,GA,EN;EN,JA
        4.14.6:MA,EP
        4.14.7:MA,GA
        4.15.1:JA,CO,TS;JV,TS
        4.15.2:JV,GA
        4.15.3:JV
        4.15.4:GA
        5.1.1
        5.1.2:EN,CM,CR,FE,JO,BO
        5.1.3:EN
        5.1.4:EN,CM,MA;MA,JV,EN
        5.1.5:EN
        5.1.6:EN,JA;EN,JA,JV
        5.1.7:EN,BO,CR,CM;EN,BO,CR,CM,GA
        5.1.8:MA,GA;EN,CM
        5.1.9:EN,JV
        5.1.10:CO
        5.1.11:EN,GA;BO,JV
        5.1.12:EN
        5.1.13:EN,CR
        5.1.14:CR,BO;BO,EN
        5.1.15:GA,CR
        5.1.16:XA,XB
        5.1.17:EN,CM,CR
        5.1.18:EN,CM,BO,CR,FE;EN,JV;JV,JA
        5.1.19:JV,JA;MA,EN
        5.1.20
        5.1.21:EN,MA,CR,BO,FE,CM,JO
        5.1.22:EN
        5.1.23:EN,GT
        5.1.24:JV,MA
        5.2.1
        5.2.2
        5.2.3:BS
        5.2.4:BS
        5.2.5
        5.2.6
        5.3.1:JV,MA
        5.3.2:JV,MA
        5.3.3:JA,TH
        5.3.4:JV,MA
        5.3.5:JV,MA
        5.3.6:JV,MA
        5.3.7:JV,MA
        5.3.8:JV,TH
        5.3.9:JV,JA
        5.3.10:JV,JA
        5.3.11:JV,JA
        5.3.12:GI,MA
        5.4.1:JA
        5.5.1:BZ
        5.5.2:MA
        5.5.3:MA,GI
        5.5.4:CO,MA,GI,JV,MG
        5.5.5:JV
        5.5.6:GI,MA,CO
        5.5.7:MA,JV
        5.5.8:MA,JV
        5.6.1:MA,GI;MA,CO;TH,AZ
        5.6.2:MA,CO,JV,GI,MG
        5.6.3:JV
        5.7.1:MA,JV;MA,JV,CO;MA,JV
        5.7.2:MA
        5.8.1:JV,CO
        5.8.2:JV,CO
        5.8.3:JV,CO
        5.8.4:JV
        5.9.1:MA,CO
        5.9.2:JV
        5.9.3:JV
        5.9.4:MA,TH;MA,CO
        5.9.5:JV,CO,MA
        5.9.6"""
