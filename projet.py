import math
import utils
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt  # for plotting
from operator import itemgetter


"""___________________________________________________________________________"""




def getPrior (train):
    s = train.query('target == 1')
    prob = len(s)/len(train)
    est_moin = prob - (1/math.sqrt(len(train)))
    est_plus = prob + (1/math.sqrt(len(train)))
    return { 'estimation' :prob,' min5pourcent' : est_moin, 'max5pourcent' : est_plus  }

""" ________________________________________________________________________________________________"""


class AbstractClassifier:
  """
  Un classifier implémente un algorithme pour estimer la classe d'un vecteur d'attributs. Il propose aussi comme service
  de calculer les statistiques de reconnaissance à partir d'un pandas.dataframe.
  """

  def ___init__(self):
    pass

  def estimClass(self, attrs):
    """
    à partir d'un dictionanire d'attributs, estime la classe 0 ou 1

    :param attrs: le  dictionnaire nom-valeur des attributs
    :return: la classe 0 ou 1 estimée
    """
    raise NotImplementedError

  def statsOnDF(self, df):
    """
    à partir d'un pandas.dataframe, calcule les taux d'erreurs de classification et rend un dictionnaire.

    :param df:  le dataframe à tester
    :return: un dictionnaire incluant les VP,FP,VN,FN,précision et rappel
    """
    raise NotImplementedError

"""________________________________________________________________________________________________"""

class APrioriClassifier(AbstractClassifier) :
    def ___init__(self):
       
        pass

    def estimClass(self, attrs):
        """
        à partir d'un dictionanire d'attributs, estime la classe 0 ou 1

        :param attrs: le  dictionnaire nom-valeur des attributs
        :return: la classe 0 ou 1 estimée
        """
        return 1
        
    def statsOnDF(self, df):
        """
        à partir d'un pandas.dataframe, calcule les taux d'erreurs de classification et rend un dictionnaire.

        :param df:  le dataframe à tester
        :return: un dictionnaire incluant les VP,FP,VN,FN,précision et rappel
        """
        vp = 0
        vn = 0
        fp = 0
        fn = 0

        for t in df.itertuples():
            dic=t._asdict()
            if dic['target'] == 1 :
                if self.estimClass(dic) == 1:
                    vp += 1
                else :
                    fn += 1
            else :
                if self.estimClass(dic) == 0 :
                    vn += 1
                else :
                    fp += 1



        precision = (vp/(vp+fp))
        rappel= (vp/(vp+fn))
        return  {'vp ': vp , 'vn ': vn ,'fp ': fp , 'fn ':fn ,'Precision' :precision , 'Rappel' : rappel }

"""_______________________________________________________________________________________________ """

def P2D_l(df,attr) :

    valeurs = df[attr].unique()

    dic1 = {}
    dic2 = {}
    for i in valeurs :
        dic1[i]=0
        dic2[i]=0
        for t in df.itertuples():
            dic=t._asdict()
            if dic['target'] == 1 and dic[attr] == i :
                dic1[i] += 1
            elif dic['target'] == 0 and dic[attr] == i: 
                dic2[i] += 1
        dic2[i] /= len(df[df["target"]==0])
        dic1[i] /= len(df[df["target"]==1])
    return { 1 : dic1 , 0 : dic2}



"""_________________________________________________________________________________________ """

def P2D_p (df,attr) :
    valeurs = df[attr].unique()

    dic1 = {}
    dic2 = {}
    dico = {}
    for i in valeurs :
        dic1[1]=0
        dic2[0]=0

        for t in df.itertuples():
            dic=t._asdict()
            if dic['target'] == 1 and dic[attr] == i :
                dic1[1] += 1
            elif dic['target'] == 0 and dic[attr] == i: 
                dic2[0] += 1
        dic2[0] /= len(df[df[attr]==i])
        dic1[1] /= len(df[df[attr]==i])
        dico[i] = { 0: dic2[0], 1 : dic1[1]}
    return dico 

"""____________________________________________________________________________________"""


class ML2DClassifier(APrioriClassifier) :

    def __init__(self,train,attr):

        self.attr = attr
        self.P2Dl = P2D_l(train,attr)
    def estimClass(self, dic):
        """
        à partir d'un dictionanire d'attributs, estime la classe 0 ou 1

        :param attrs: le  dictionnaire nom-valeur des attributs
        :return: la classe 0 ou 1 estimée
        """
        if  self.P2Dl[1][dic[self.attr]] > self.P2Dl[0][dic[self.attr]]:
            return 1
        return 0

"""___________________________________________________________________________________________"""

class MAP2DClassifier(APrioriClassifier) :

    def __init__(self,train,attr):

        self.attr = attr
        self.P2Dp = P2D_p(train,attr)
    def estimClass(self, dic):
        """
        à partir d'un dictionanire d'attributs, estime la classe 0 ou 1

        :param attrs: le  dictionnaire nom-valeur des attributs
        :return: la classe 0 ou 1 estimée
        """


        if  self.P2Dp[dic[self.attr]][1] > self.P2Dp[dic[self.attr]][0]:
            return 1
        return 0

"""____________________________________________________________________________________"""

def nbParams(data,attr=None) :


    if attr == None :
        attr = data.columns
    valeur = 1
    for col in attr :
        valeur *= (len(data[col].unique()))

    print(len(attr),"Variables coute : ", valeur*8)

    return valeur*8

"""_______________________________________________________________________________________"""

def nbParamsIndep(data,attr=None) :

    if attr == None :
        attr = data.columns

    valeur = 0
    for col in attr :
        valeur += (len(data[col].unique()))*8

    print(len(attr),"Variables coute : ", valeur)

    return valeur

"""_____________________________________________________________________________________________"""

def drawNaiveBayes(df,target):
    attr = df.columns
    graph =""
    for i in attr :
        graph = graph+target+"->"+i+";"
    return utils.drawGraph(graph)

"""_______________________________________________________________________________________________"""


def nbParamsNaiveBayes (data,target,attr=None) :

    if attr == None :
        attr = data.columns

    valeur = 16
    for col in attr[1:] :
        valeur += (len(data[col].unique()))*16

    print(len(attr),"Variables coute : ", valeur)

"""___________________________________________________________________________________"""


class MLNaiveBayesClassifier (APrioriClassifier) :

    def __init__(self,data):

        self.attr = list(data.columns)
        self.attr.remove('target')
        self.P2Dl = []

        for col in self.attr:
            if col != 'target':
                self.P2Dl.append(P2D_l(data,col))

    def estimClass(self,dic):

        cla1 =1
        cla2 =1

        for i in range(len(self.attr)):
            if (dic[self.attr[i]] in self.P2Dl[i][0])  :
                cla1 *= self.P2Dl[i][1][dic[self.attr[i]]]
                cla2 *= self.P2Dl[i][0][dic[self.attr[i]]]
            else :
                cla1 =0
                cla2 =0

        if cla1 > cla2 :
            return 1
        return 0 

"""____________________________________________________________________________________________"""

class MAPNaiveBayesClassifier (APrioriClassifier) :

    def __init__(self,data):

        self.cla1 = getPrior(data)['estimation']
        self.cla2 = 1 - self.cla1
        self.attr = list(data.columns)
        self.attr.remove('target')
        self.P2Dl = {}

        for col in self.attr:
            if col != 'target':
                self.P2Dl[col]=P2D_l(data,col)

    def estimClass(self,dic):


        cla1 =self.cla1
        cla2 =self.cla2

        for col in self.attr:
            if (dic[col] in self.P2Dl[col][0])  :

                cla1 *= self.P2Dl[col][1][dic[col]]
                cla2 *= self.P2Dl[col][0][dic[col]]
            else :
                cla1 =0
                cla2 =0

        if cla1 > cla2 :
            return 1
        return 0 
        
"""_______________________________________________________________________________________________"""


def isIndepFromTarget(df,attr,x):
    """
    Vérifie si attr est indépendant de target au seuil de x%.
    
    :param df: dataframe. Doit contenir une colonne appelée "target" ne contenant que 0 ou 1.
    :param attr: le nom d'une colonne du dataframe df.
    :param x: seuil de confiance.
    """
    list_val = np.unique(df[attr].values) # Valeurs possibles de l'attribut.
    dico_val = {list_val[i]: i for i in range(list_val.size)} 
    #un dictionnaire associant chaque valeur a leur indice en list_val.
    
    mat_cont = np.zeros((2, list_val.size), dtype = int)
    
    for i, row in df.iterrows():
        j =  row[attr]
        mat_cont[row["target"], dico_val[j]]+= 1 
    
    _, p, _, _ = scipy.stats.chi2_contingency(mat_cont)
    return p > x

"""______________________________________________________________________________________________"""

class ReducedMAPNaiveBayesClassifier(APrioriClassifier) :

    def __init__(self,data,seuil):

        self.data = data
        self.cla1 = getPrior(data)['estimation']
        self.cla2 = 1 - self.cla1
        self.attr = list(data.columns)
        self.attr.remove('target')
        self.P2Dl = {}

    
        for col in list(data.columns):
            if  col != 'target' and  not isIndepFromTarget(data,col,seuil):
                self.P2Dl[col]=P2D_l(data,col)
            else :
                if col != 'target' :
                    self.attr.remove(col)


    def estimClass(self,dic):

        cla1 =self.cla1
        cla2 =self.cla2

        for col in self.attr:

            if (dic[col] in self.P2Dl[col][0])  :
                cla1 *= self.P2Dl[col][1][dic[col]]
                cla2 *= self.P2Dl[col][0][dic[col]]
            else :
                cla1 =0
                cla2 =0

        if cla1 > cla2 :
            return 1
        return 0 

    def draw(self) :
        return drawNaiveBayes(self.data[self.attr],'target')

"""__________________________________________________________________________________________"""

class ReducedMLNaiveBayesClassifier(APrioriClassifier):

    def __init__(self,data,seuil):

        self.data = data
        self.attr = list(data.columns)
        self.attr.remove('target')
        self.P2Dl = {}

    
        for col in list(data.columns):
            if  col != 'target' and  not isIndepFromTarget(data,col,seuil):
                self.P2Dl[col]=P2D_l(data,col)
            else :
                if col != 'target' :
                    self.attr.remove(col)


    def estimClass(self,dic):

        cla1 = 1
        cla2 = 1

        for col in self.attr:

            if (dic[col] in self.P2Dl[col][0])  :
                cla1 *= self.P2Dl[col][1][dic[col]]
                cla2 *= self.P2Dl[col][0][dic[col]]
            else :
                cla1 =0
                cla2 =0

        if cla1 > cla2 :
            return 1
        return 0 

    def draw(self) :
        return drawNaiveBayes(self.data[self.attr],'target')       

"""__________________________________________________________________________________________________________"""

def mapClassifiers(dic,df):
    pres= []
    rapp= []
    for dico in dic:
        pres.append(dic[dico].statsOnDF(df)['Precision'])
        rapp.append(dic[dico].statsOnDF(df)['Rappel'])

    fig, ax = plt.subplots()
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.set_xlabel("Précision")
    ax.set_ylabel("Rappel")
    ax.scatter(pres, rapp, marker = 'x', c = 'red') 
    for i, nom in enumerate(dic):
        ax.annotate(nom, (pres[i], rapp[i]))
    
    plt.show()


"""_______________________________________________________________________________________________________________"""

def MutualInformation(df,x,y):
    res = 0
    for i in df[x].unique():
        px = 0
        py = 0
        pxy = 0
        for j in df[y].unique():
            px = len(df[df[x]==i])/len(df)
            py = len(df[df[y]==j])/len(df)
            pxy = len(df[df[x]==i][df[y]==j])/len(df)
            if ((pxy)/(px*py)) >0 :
                res += pxy*math.log2(((pxy)/(px*py)))
    return res

"""____________________________________________________________________________________________________________"""

def ConditionalMutualInformation(df,x,y,z):
    res = 0.0
    for i in df[x].unique():
        pxz = 0
        pyz = 0
        pz = 0
        pxyz = 0
        for j in df[y].unique():
            for k in df[z].unique():
                pz = len(df[df[z]==k])/len(df)
                pxz = len(df[df[x]==i][df[z]==k])/len(df)
                pyz = len(df[df[y]==j][df[z]==k])/len(df)
                pxyz = len(df[df[x]==i][df[y]==j][df[z]==k])/len(df)
                if pxz !=0 and pyz !=0 and (pz*pxyz/(pxz*pyz)) >0  :
                    res += pxyz * math.log2((pz*pxyz/(pxz*pyz)))
    return res

"""_____________________________________________________________________________________________________"""

def MeanForSymetricWeights(a):
    res = 0
    for i in range(len(a)):
        for j in range(len(a[0])):
            res += a[i][j]
    return res/((len(a)*len(a[0]))-len(a))

"""____________________________________________________________________________________________________ """

def SimplifyConditionalMutualInformationMatrix(cmis) :
    moy = MeanForSymetricWeights(cmis)

    for i in range(len(cmis)):
        for j in range(len(cmis[0])):
            if cmis[i][j] < moy :
                cmis[i][j]= 0.0

"""______________________________________________________________________________________________________"""

def Kruskal(df,matrice):
    SimplifyConditionalMutualInformationMatrix(matrice)
    graph = []
    kruskal = []
    groups = {}
    liste = df.columns

    for i in range(len(matrice)):
        for j in range(len(matrice)):
            if matrice[j][i] != 0 :
                graph.append([j,i,matrice[i][j]])
    graph = list(sorted(graph, key=itemgetter(2)))
    graph.reverse()
    
    for x , y ,poid in graph:
        if( not find(x,y,groups)):
            kruskal.append((liste[x],liste[y],poid))
            union(x,y,groups)
    return kruskal


"""______________________________________________________________________________________________________"""

def union(x,y,groups):

    if not x in groups and not y in groups:
        groups[x] =y
        groups[y] =y
    elif not x in groups :
        if not y in groups :
            groups[x]=y
            groups[y]=y
        else:
            groups[x]=groups[y]
    elif not y in groups:
        groups[y]=groups[x]
    else:
        for key in groups :
            if groups[key]==groups[x]:
                groups[key] = groups[y]



"""________________________________________________________________________________________________________"""


def find (x,y,groups):
    if not x in groups or not y in groups :
        return False
    return groups[x] == groups[y]

"""___________________________________________________________________________________________________________"""

def ConnexSets(list_arcs):
    groups = {}
    res = []
    for x , y ,poid in list_arcs:
        union(x,y,groups)
    for valeur in groups.values():
        set_ = []
        for key in groups :
            if groups[key]== valeur :
                set_.append(key)
        if not set(set_) in res:
            res.append(set(set_))
    return res

"""___________________________________________________________________________________________________________"""

def OrientConnexSets(df, arcs, classe):
    """
    Utilise l'information mutuelle (entre chaque attribut et la classe) pour
    proposer pour chaque ensemble d'attributs connexes une racine et qui rend 
    la liste des arcs orientés.
    
    :param df: Dataframe contenant les données. 
    :param arcs: liste d'ensembles d'arcs connexes.
    :param classe: colonne de réference dans le dataframe pour le calcul de 
    l'information mutuelle.
    """
    arcs_copy = arcs.copy()
    list_sets = ConnexSets(arcs_copy)
    list_arbre = []
    for s in list_sets:
        col_max = ""
        i_max = -float("inf") 
        for col in s:
            i = MutualInformation(df, col, classe)
            if i > i_max:
                i_max = i
                col_max = col
        list_arbre += creeArbre(arcs_copy, col_max)
    return list_arbre

"""_________________________________________________________________________________________________________"""
 
def creeArbre(arcs, racine): 
    """
    À partir d'une liste d'arcs et d'une racine, renvoie l'arbre orienté depuis
    cette racine. La liste arcs est modifié par cette fonction.
    
    :param arcs: liste d'ensembles d'arcs connexes.
    :param racine: nom d'un sommet.
    """
    res = []
    file = [racine]
    while file != []:
        sommet = file.pop(0)
        arcs_copy = arcs.copy()
        for (u, v, poids) in arcs_copy:
            if sommet == u:
                res.append((u, v))
                arcs.remove((u, v, poids))
                file.append(v)
            elif sommet == v:
                res.append((v, u))
                arcs.remove((u, v, poids))
                file.append(u)
    return res 