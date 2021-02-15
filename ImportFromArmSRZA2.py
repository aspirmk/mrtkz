# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 13:29:59 2020

@author: aspirmk

Преобразование моделей АРМ СРЗА в формате Excel в скрипты Python3 для МРТКЗ 
"""
import xlrd
import numpy as np


a = {'А': 'A', 'Б': 'B', 'В': 'V', 'Г': 'G', 'Д': 'D', 'Е': 'E', 'Ё': 'Ey',
     'Ж': 'Zh', 'З': 'Z', 'И': 'I', 'Й': 'Iy', 'К': 'K', 'Л': 'L', 'М': 'M',
     'Н': 'N', 'О': 'O', 'П': 'P', 'Р': 'R', 'С': 'S', 'Т': 'T', 'У': 'U',
     'Ф': 'F', 'Х': 'Kh', 'Ц': 'Tc', 'Ч': 'Ch', 'Ш': 'Sh', 'Щ': 'Shch',
     'Ы': 'Y', 'Э': 'Ee', 'Ю': 'Iu', 'Я': 'Ia'}

def translite(str):
    res = ['q_']
    for ch in str:
        if ch in a.keys():
            ch = a[ch]
        res.append(ch)
    return ''.join(res)


class ImpQ:
    def __init__(self, name, desc, key):
        self.name = name
        if isinstance(name, str):
            self.tlname = translite(name)
        else:
            self.tlname = 'q_'+str(name)
        if isinstance(desc, str):
            desc = desc.rstrip()
        self.desc = desc
        self.key = key
        self.elem = None
        self.plist = []

    def addp(self, p):
        self.plist.append(p)


class ImpP:
    def __init__(self, typ, par, q1, q2, Nel, Z1, Z2, Z0, EKB1, F1L, KB0):
        self.typ = typ
        self.par = par
        tlq1 = '0'
        q1name = '0'
        self.q1 = q1
        tlq2 = '0'
        q2name = '0'
        self.q2 = q2
        if isinstance(q1, ImpQ):
            tlq1 = q1.tlname[2:]
            q1name = q1.name
        if isinstance(q2, ImpQ):
            tlq2 = q2.tlname[2:]
            q2name = q2.name
        self.name = '{} {}-{}'.format(par, q1name, q2name)
        self.tlname = 'p_'+str(par)+'_'+tlq1+'_'+tlq2
        self.Z1 = Z1
        self.Z2 = Z2
        self.Z0 = Z0
        self.EKB1 = EKB1
        self.F1L = F1L
        self.KB0 = KB0
        self.elem = Nel


class ImpE:
    def __init__(self, name, desc):
        self.name = name
        if isinstance(desc, str):
            desc = desc.rstrip()
        self.desc = desc
        self.qlist = []
        self.plist = []
        self.builded = False

    def addp(self, p):
        self.plist.append(p)

    def addq(self, q):
        self.qlist.append(q)

    def build(self):
        strres = ''
        if not self.builded:
            res = []
            self.builded = True
            res.append("\n")
            res.append("\n")
            res.append("\n")
            res.append("#|---------------------------------------------------------------\n")
            res.append("#| ЭЛЕМЕНТ: " + str(self.name) + " - " + self.desc + '\n')
            res.append("#|---------------------------------------------------------------\n")
            res.append("\n")
            res.append("\n")
            res.append("#УЗЛЫ\n")
            for kq in self.qlist:
                res.append("\n")
                res.append("{} = mrtkz.Q(mdl, '{}', desc='{}')\n".format(kq.tlname, kq.name, kq.desc))
            res.append("\n")
            res.append("\n")
            res.append("#ВЕТВИ\n")
            for kp in self.plist:
                tlq1 = '0'
                if isinstance(kp.q1, ImpQ):
                    tlq1 = kp.q1.tlname
                tlq2 = '0'
                if isinstance(kp.q2, ImpQ):
                    tlq2 = kp.q2.tlname
                res.append("\n")
                if kp.typ == 0: #Простая ветвь
                    res.append("{} = mrtkz.P(mdl, '{}', {}, {}, ({}, {}, {}))\n".format(kp.tlname, kp.name, tlq1, tlq2, kp.Z1, kp.Z2, kp.Z0))
                elif kp.typ == 1: #Выключатель включенный
                    res.append("{} = mrtkz.P(mdl, '{}', {}, {}, ({}, {}, {}))\n".format(kp.tlname, kp.name, tlq1, tlq2, kp.Z1, kp.Z2, kp.Z0))
                elif kp.typ == 101: #Выключатель отключенный
                    res.append("#{} = mrtkz.P(mdl, '{}', {}, {}, ({}, {}, {}))\n".format(kp.tlname, kp.name, tlq1, tlq2, kp.Z1, kp.Z2, kp.Z0))
                elif kp.typ == 3: #Трансформатор
                    res.append("{} = mrtkz.P(mdl, '{}', {}, {}, ({}, {}, {}), T=({}, 0))\n".format(kp.tlname, kp.name, tlq1, tlq2, kp.Z1, kp.Z2, kp.Z0, kp.EKB1))
                elif kp.typ == 4: #Система или Генератор
                    if kp.F1L == 0:#Фаза Э.Д.С. равна нулю
                        res.append("{} = mrtkz.P(mdl, '{}', {}, {}, ({}, {}, {}), E=({}/1.732, 0, 0))\n".format(kp.tlname, kp.name, tlq1, tlq2, kp.Z1, kp.Z2, kp.Z0, 1000*kp.EKB1))
                    else:#Фаза Э.Д.С. не равна нулю
                        res.append("{} = mrtkz.P(mdl, '{}', {}, {}, ({}, {}, {}), E=({}/1.732*np.exp(1j*np.pi/180*{}), 0, 0))\n".format(kp.tlname, kp.name, tlq1, tlq2, kp.Z1, kp.Z2, kp.Z0, kp.EKB1, kp.F1L))
                elif kp.typ == 5: #Ветвь с B
                    res.append("{} = mrtkz.P(mdl, '{}', {}, {}, ({}, {}, {}), B=({}, {}, {}))\n".format(kp.tlname, kp.name, tlq1, tlq2, kp.Z1, kp.Z2, kp.Z0, kp.EKB1*1e-6j, kp.EKB1*1e-6j, kp.KB0*1e-6j))
            strres = ''.join(res)
        return strres


class ImpM:
    def __init__(self, N):
        self.N = N
        self.M = np.zeros((N,N),dtype=np.cdouble)
        self.plist = []


class ImpModel:
    def __init__(self, name, desc):
        self.name = name
        self.desc = desc
        self.qlist = {}
        self.plist = {}
        self.mlist = []
        self.elist = {}
        self.el0 = ImpE(0,'Общий элемент')
        self.elist[0] = self.el0


    def ImpFromXLS(self, filename):
        workbook = xlrd.open_workbook(filename)
        # Load a specific sheet by name
        QTable = workbook.sheet_by_name('Наим.узлов')
        MTable = workbook.sheet_by_name('Индуктивные группы')
        PTable = workbook.sheet_by_name('Таблица ветвей')
        ETable = workbook.sheet_by_name('Наим.элементов')
        NQ = int(QTable.cell(0, 0).value[28:-1])
        NP = int(PTable.cell(0, 0).value[18:-1])
        NM = int(MTable.cell(0, 0).value[27:-1])
        NE = int(ETable.cell(0, 0).value[32:-1])
        print('Кол-во узлов - ', NQ)
        print('Кол-во ветвей - ', NP)
        print('Кол-во индуктивных групп - ', NM)
        print('Кол-во элементов - ', NE)
        for rowi in range(2,NE+2):
            ElName = int(ETable.cell(rowi, 0).value)
            ElDesc = ETable.cell(rowi, 1).value
            ke = ImpE(ElName, ElDesc)
            self.elist[ElName] = ke
        for rowi in range(2,NQ+2):
            QName = QTable.cell(rowi, 0).value
            if isinstance(QName, float):
                QName = int(QName)
            else:
                QName = QName.rstrip()
            QDesc = QTable.cell(rowi, 1).value
            QKey = int(QTable.cell(rowi, 2).value)
            kq = ImpQ(QName, QDesc, QKey)
            self.qlist[QName] = kq

        for rowi in range(2,NP+2):
            q1 = 0
            q2 = 0

            PTyp = int(PTable.cell(rowi, 0).value)
            PPar = int(PTable.cell(rowi, 1).value)

            PQ1 = PTable.cell(rowi, 2).value
            if isinstance(PQ1, float):
                PQ1 = int(PQ1)
            else:
                PQ1 = PQ1.rstrip()
            if PQ1 != 0:
                q1 = self.qlist[PQ1]

            PQ2 = PTable.cell(rowi, 3).value
            if isinstance(PQ2, float):
                PQ2 = int(PQ2)
            else:
                PQ2 = PQ2.rstrip()
            if PQ2 != 0:
                q2 = self.qlist[PQ2]

            PEl = self.elist[int(PTable.cell(rowi, 4).value)]

            PZ1 = PTable.cell(rowi, 5).value + 1j * PTable.cell(rowi, 6).value
            PZ2 = PTable.cell(rowi, 12).value + 1j * PTable.cell(rowi, 13).value
            if PZ2 == 0:
                PZ2 = PZ1
            PZ0 = PTable.cell(rowi, 9).value + 1j * PTable.cell(rowi, 10).value
            if not PTyp in (1, 101) and PZ0 == 0:
                 PZ0 = PZ1
            PEKB1 = PTable.cell(rowi, 7).value
            PF1L = PTable.cell(rowi, 8).value
            PKB0 = PTable.cell(rowi, 11).value
            kp = ImpP(PTyp, PPar, q1, q2, PEl, PZ1, PZ2, PZ0, PEKB1, PF1L, PKB0)
            PEl.addp(kp)
            if isinstance(q1, ImpQ):
                q1.addp(kp)
            if isinstance(q2, ImpQ):
                q2.addp(kp)
            keyp = '{} {}-{}'.format(PPar, PQ1, PQ2)
            self.plist[keyp] = kp

        for kq in self.qlist.values():
            if kq.plist:
                kq.elem = kq.plist[0].elem
                for kp in kq.plist[1:]:
                    if not kq.elem is kp.elem:
                        kq.elem = self.el0
                        self.el0.addq(kq)
                        break
                else: kq.elem.addq(kq)

        ijk = 1
        for ij in range(NM):
            strMN = MTable.cell(ijk, 0).value
            kmN = int(strMN[12:])
            km = ImpM(kmN)
            for ij1 in range(kmN):
                rowi = ijk+2+ij1
                PPar = int(MTable.cell(rowi, 0).value)
                PQ1 = MTable.cell(rowi, 1).value
                if isinstance(PQ1, float):
                    PQ1 = int(PQ1)
                else:
                    PQ1 = PQ1.rstrip()
                PQ2 = MTable.cell(rowi, 2).value
                if isinstance(PQ2, float):
                    PQ2 = int(PQ2)
                else:
                    PQ2 = PQ2.rstrip()
                keyp = '{} {}-{}'.format(PPar, PQ1, PQ2)
                km.plist.append(self.plist[keyp])
                for ij2 in range(kmN):
                    coli = 3 + 2*ij2
                    km.M[ij1,ij2] = MTable.cell(rowi, coli).value + 1j * MTable.cell(rowi, coli+1).value

            self.mlist.append(km)
            ijk += 2 + kmN


    def Exp2MRTKZ(self, filename, RW=False):
#        if not RW:
#            with open(filename+'.py', "r") as file:
#                raise ValueError('Файл с таким именем уже существует')
        with open(filename+'.py', "wb") as file:
            res = []
            res.append("# -*- coding: utf-8 -*-\n")
            res.append("'''Импортирование модуля расчета ТКЗ (mrtkz3.py) и\n")
            res.append("модуля расчета параметров воздушных линий  PVL,\n")
            res.append("которые должны находиться в той же папке, где и настоящий файл'''\n")
            res.append("#import PVL5 as PVL\n")
            res.append("import mrtkz3 as mrtkz\n")
            res.append("#import numpy as np\n")
            res.append("\n")
            res.append("\n")
            res.append("#||==============================================================\n")
            res.append("#|| " + self.name + " - " + self.desc + '\n')
            res.append("#||==============================================================\n")
            res.append("\n")
            res.append("\n")
            res.append("\n")
            res.append("#Создание расчетной модели\n")
            res.append("mdl=mrtkz.Model()\n")
            res.append("\n")
            res.append("\n")
            res.append(self.elist[0].build())
            res.append("\n")
            res.append("\n")
            for ij,km in enumerate(self.mlist):
                for kp in km.plist:
                    res.append(kp.elem.build())
                res.append("\n")
                res.append("\n")
                res.append("#|---------------------------------------------------------------\n")
                res.append("#| Взаимоиндуктивность нулевой последовательности №{}\n".format(ij))
                res.append("#|---------------------------------------------------------------\n")
                for ij1 in range(km.N):
                    for ij2 in range(ij1):
                        res.append("\n")
                        res.append("mrtkz.M(mdl, '{}  --  {}', {}, {}, {}, {})\n".format(km.plist[ij1].name, km.plist[ij2].name, km.plist[ij1].tlname, km.plist[ij2].tlname, km.M[ij1, ij2], km.M[ij1, ij2]))
            res.append("\n")
            res.append("\n")
            res.append("#Проверка на вырожденность\n")
            res.append("mdl.Test4Singularity()\n")
            res.append("\n")
            res.append("#Создание однофазного КЗ\n")
            res.append("#KZ1 = mrtkz.N(mdl,'Однофазное КЗ', q_, 'A0')\n")
            res.append("\n")
            res.append("#Формирование разреженной СЛАУ и расчет электрических параметров\n")
            res.append("#mdl.Calc()\n")
            res.append("\n")
            res.append("#Вывод таблицы результатов расчетов по КЗ\n")
            res.append("#KZ1.res()\n")

            for ke in self.elist.values():
                res.append(ke.build())
            strres = ''.join(res)
            file.write(strres.encode('utf-8'))


if __name__ == '__main__':    
    #Путь к файлу Excel, содержащего модель АРМ СРЗА
    input_filename = 'тест4.xls'    
    
    #Название выходного файла скрипта сконвертированной модели в Python3, 
    #разрешение файла .py добавляется автоматически
    output_filename = 'test4'
    
    mdl = ImpModel('Тест',"Тестовая модель")
    mdl.ImpFromXLS(input_filename)
    mdl.Exp2MRTKZ(output_filename, RW=True)