# -*- coding: utf-8 -*-
'''
МОДУЛЬ РАСЧЕТА ТОКОВ КОРОТКОГО ЗАМЫКАНИЯ (М Р Т К З)

Версия 3.10

г.Саратов 28.11.2020

История изменений

28.11.2020
1. В классе Model добавлен метод ImportFromPVL(PVL_Sech), предназначенный для
   импорта результатов расчетов параметров схемы замещения ВЛ с помощью модуля PVL
   mdl.ImportFromPVL(PVL_Sech)
   где PVL_Sech - ссылка на сечение (объект класса sech модуля PVL)

18.11.2020
1. Добавлены описания к функциям, классам и методам МРТКЗ,
    в том числе дано подробное описание алгоритма метода mdl.Calc()
2. Добавлены специальные нeсимметрии вида:
    - КЗ по нулевой последовательности - 'N0' для моделирования заземления нейтрали за
      трансформатором Yg/D или за соответствующей парой обмоток тр-ра
    - Обрыв по нулевой последовательности - 'N0' для моделирования сети с изолированной
      нейтралью, устанавливается на ветви разделяющей сеть с глухо или эффективно
      заземленной нейтралью и сетью с изолированной нейтралью
3. Выверено моделирование ветвей с поперечной емкостной проводимостью B,
    для этого если pl62w+ или аналогичное ПО выдает (Например)
    B1 = В2 = 90 мкСм (1/Ом*10^-6), B0 = 60 мкСм (1/Ом*10^-6)
    то при создании ветви надо заполнять параметры емкостной проводимости
    B=(90e-6j,90e-6j,60e-6j)
4. Выверено моделирование трансформаторных ветвей
    T=(Ktrans,GrT) - безразмерные параметры трансформаторной ветви:
    Ktrans - коэффициент трансформации силового трансформатора, например 115/11
    GrT - группа обмоток обмотки подключенной к узлу 2 (от 0 до 11)
    Так например для трансформатора с номинальными напряжениями обмоток 115 и 10,5 кВ
    и схемой соединения обмоток Y/D-11 надо заполнять параметры трансформаторной ветви
    T=(115/10.5,11)
5. Добавлены новые методы класса Model
    - AddNQ для группового добавления узлов (сечения узлов)
    mdl.AddNQ(NQ,Nname)
    где NQ - количество создаваемых узлов
        Nname - общее имя узлов, к уникальному имени узлу будет добавляться
        номер из сечения от 1 до NQ
    - AddNP для группового добавления ветвей (сечения ветвей)
    в том числе с поперечной емкостной проводимостью B и
    взаимоиндукцией нулевой последовательности
    mdl.AddNP(self,Nname,listq1,listq2,Z12,Z0) - без учета емкостной проводимости
    mdl.AddNP(self,Nname,listq1,listq2,Z12,Z0,B12,B0) - с учетом емкостной проводимости
    где listq1 и listq2 - сечения (списки) узлов
        Nname - общее имя сечения ветвей, к уникальному имени ветви будет добавляться
        номер из сечения от 1 до N, где N - количество создаваемых ветвей
        Z12 - вектор numpy.ndarray значений сопротивлений ветвей прямой/обратной последовательности
        Z0 - квадратная матрица numpy.ndarray значений сопротивлений ветвей и взаимоиндукций нулевой последовательности
        B12 - вектор numpy.ndarray значений поперечной емкостной проводимости прямой/обратной последовательности
        B0 - квадратная матрица numpy.ndarray значений поперечной емкостной проводимости нулевой последовательности
'''

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
#import PVL5 as PVL

Kf=-1j*np.pi/6
r2d=180/np.pi
a=0.5*(-1+1j*np.sqrt(3))
a2=0.5*(-1-1j*np.sqrt(3))
a0=1.0+0.0j
vA=[a0,a0,a0]
v_A=[-a0,-a0,-a0]
vB=[a2,a,a0]
vC=[a,a2,a0]


class Q:
    '''
    Класс трехфазного электрического узла, необходим для формирования расчетной
    модели и получения результатов расчета

    Создание узла с помощью конструктора
    Q(model,name,desc='')
    Q(model,name)
    где:
       model - объект расчетной модели в которой создается узел
       name - краткое название узла, обращение к узлу по его имени не предусмотрено
       desc - Примечание или любая другая текстовая информация, можно не задавать.
    Результатом конструктора узла является объект узла, который используется для
    формирования расчетной модели и вывода результатов расчетов

    Пользовательские функции для объекта узла q
    Вывод на экран параметров узла - его номера и названия
    q.par()

    Вывод сводной таблицы результатов расчетов для узла q
    q.res()

    Вывод конкретного параметра ParName в виде компексного числа
    для последующего использования в расчетах
    q.res(ParName)
    где ParName может принимать значения:
    'U1','U2','U0','3U0','UA','UB','UC','UABC','UAB','UBC','UCA','UAB_BC_CA'

    Вывод конкретного параметра ParName в заданной форме Form:
    q.res(ParName,Form)
    где Form может принимать значения
    'R' - Активная составляющая
    'X' - Реактивная составляющая
    'M' - Модуль комплексного числа
    '<f' - Фаза вектора в градусах
    'R+jX' - Текстовый вид комплексного числа
    'M<f' - Текстовый вид комплексного числа

    Еще один способ получения конкректного параметра результата в виде
        компексного числа для его последующего использования в расчетах
        q.ParName
        где ParName может принимать значения:
        U1,U2,U0,UA,UB,UC,UABC,UAB,UBC,UCA,UAB_BC_CA
    '''
    def __init__(self,model,name,desc=''):
        ''' Конструктор объекта узла
        Q(model,name,desc='')
        Q(model,name)
        где:
           model - объект расчетной модели в которой создается узел
           name - краткое название узла, обращение к узлу по его имени не предусмотрено
           desc - Примечание или любая другая текстовая информация, можно не задавать.
        Результатом конструктора узла является объект узла, который используется для
        формирования расчетной модели и вывода результатов расчетов
        '''
        if not isinstance(model, Model):
            raise TypeError('Ошибка при добавлении узла -', name, '\n',
                            'Аргумент model должен иметь тип Model!')
        model.nq+=1
        model.bq.append(self)
        self.id=model.nq
        self.model=model
        self.name=name
        self.desc=desc
        self.plist=[]
        self.kn=None

    def addp(self,kp):
        '''Служебный метод, предназачен для информирования узла о подключенных к нему ветвей'''
        self.plist.append(kp)

    def update(self):
        '''Служебный метод, предназачен для проверки информации о подключенных к узлу ветвей'''
        temp_plist = self.plist
        self.plist=[]
        for kp in temp_plist:
            if (kp.q1 is self) or (kp.q2 is self):
                self.plist.append(kp)

    def setn(self,kn):
        '''Служебный метод, предназачен для информирования узла о наличии КЗ в данном узле'''
        self.kn=kn

    def par(self):
        '''Вывод на экран параметров узла - его номера и названия'''
        print('Узел №', self.id, ' - ', self.name)

    def getres(self):
        '''Служебный метод, возвращает результат расчета по данному узлу -
        напряжения прямой, обратной и нулевой последовательностей, тоже что и q.res('U120')'''
        if self.model is None:
            raise ValueError('Ошибка при выводе результатов расчетов Узла №', self.id, ' - ', self.name, '\n',
                            'Узел не принадлежит какой либо модели!')
        if self.model.X is None:
            raise ValueError('Ошибка при выводе результатов расчетов Узла №', self.id, ' - ', self.name, '\n',
                            'Не произведен расчет электрических величин!')
        qId = 3*(self.model.np+self.id-1)
        return self.model.X[qId:qId+3]

    def res(self,parname='',subpar=''):
        '''Вывод сводной таблицы результатов расчетов для узла q
        q.res()

        Вывод конкретного параметра ParName в виде компексного числа
        для последующего использования в расчетах
        q.res(ParName)
        где ParName может принимать значения:
        'U1','U2','U0','3U0','UA','UB','UC','UABC','UAB','UBC','UCA','UAB_BC_CA'

        Вывод конкретного параметра ParName в заданной форме Form:
        q.res(ParName,Form)
        где Form может принимать значения
        'R' - Активная составляющая
        'X' - Реактивная составляющая
        'M' - Модуль комплексного числа
        '<f' - Фаза вектора в градусах
        'R+jX' - Текстовый вид комплексного числа
        'M<f' - Текстовый вид комплексного числа'''
        u1,u2,u0 = self.getres()
        if parname=='':
            uA,uB,uC,uAB,uBC,uCA = msymm2faze(u1,u2,u0)
            print('Узел № {} - {}'.format(self.id, self.name))
            print("U1  = {0:>7.0f} < {1:>6.1f} | U2  = {2:>7.0f} < {3:>6.1f} | 3U0 = {4:>7.0f} < {5:>6.1f}".format(np.abs(u1),r2d*np.angle(u1),np.abs(u2),r2d*np.angle(u2),np.abs(3*u0),r2d*np.angle(u0)))
            print("UA  = {0:>7.0f} < {1:>6.1f} | UB  = {2:>7.0f} < {3:>6.1f} | UC  = {4:>7.0f} < {5:>6.1f}".format(np.abs(uA),r2d*np.angle(uA),np.abs(uB),r2d*np.angle(uB),np.abs(uC),r2d*np.angle(uC)))
            print("UAB = {0:>7.0f} < {1:>6.1f} | UBC = {2:>7.0f} < {3:>6.1f} | UCA = {4:>7.0f} < {5:>6.1f}".format(np.abs(uAB),r2d*np.angle(uAB),np.abs(uBC),r2d*np.angle(uBC),np.abs(uCA),r2d*np.angle(uCA)))
        else:
            res = mselectz[parname]([u1,u2,u0],[0j,0j,0j])
            if isinstance(res, (tuple, list)):
                return mform3[subpar](res,parname)
            else:
                return mform1[subpar](res,parname)

    def __getattr__(self, attrname):
        '''
        Еще один способ получения конкректного параметра результата в виде
        компексного числа для его последующего использования в расчетах
        q.ParName
        где ParName может принимать значения:
        U1,U2,U0,UA,UB,UC,UABC,UAB,UBC,UCA,UAB_BC_CA
        '''
        u1,u2,u0 = self.getres()
        res = mselectz[attrname]([u1,u2,u0],[0j,0j,0j])
        return res

    def __repr__(self):
        '''
        Еще один способ вывода сводной таблицы результатов расчетов для узла q
        В командной строке интерпретара набрать название переменной объекта узла и нажать Enter
        q Enter
        '''
        u1,u2,u0 = self.getres()
        uA,uB,uC,uAB,uBC,uCA = msymm2faze(u1,u2,u0)
        strres  = "Узел № {} - {}\n".format(self.id, self.name)
        strres += "U1  = {0:>7.0f} < {1:>6.1f} | U2  = {2:>7.0f} < {3:>6.1f} | 3U0 = {4:>7.0f} < {5:>6.1f}\n".format(np.abs(u1),r2d*np.angle(u1),np.abs(u2),r2d*np.angle(u2),np.abs(3*u0),r2d*np.angle(u0))
        strres += "UA  = {0:>7.0f} < {1:>6.1f} | UB  = {2:>7.0f} < {3:>6.1f} | UC  = {4:>7.0f} < {5:>6.1f}\n".format(np.abs(uA),r2d*np.angle(uA),np.abs(uB),r2d*np.angle(uB),np.abs(uC),r2d*np.angle(uC))
        strres += "UAB = {0:>7.0f} < {1:>6.1f} | UBC = {2:>7.0f} < {3:>6.1f} | UCA = {4:>7.0f} < {5:>6.1f}".format(np.abs(uAB),r2d*np.angle(uAB),np.abs(uBC),r2d*np.angle(uBC),np.abs(uCA),r2d*np.angle(uCA))
        return (strres)



class P:
    '''
    Класс трехфазной ветви, необходим для формирования расчетной модели
    и получения результатов расчета

    Создание ветви с помощью конструктора
    P(model,name,q1,q2,Z) - простая ветвь
    P(model,name,q1,q2,Z,desc='Примечание') - ветвь с текстовым примечанием
    P(model,name,q1,q2,Z,E=(E1,E2,E0)) - ветвь представляющая энергосистему, генератор (Вольт - фазные)
    P(model,name,q1,q2,Z,B=(B1,B2,B0)) - ветвь c наличием поперечной емкостной проводимостью B/2 (См)
    P(model,name,q1,q2,Z,T=(Ktrans,GrT)) - ветвь представляющая трансформатор
    где:
       model - объект расчетной модели в которой создается ветвь
       name - краткое название ветви, обращение к ветви по ее имени не предусмотрено
       q1,q2 - число 0, что означает подключение ветви соответствующим концом к земле,
               объект узла принадлежащего той же расчетной модели
       desc - Примечание или любая другая текстовая информация, можно не задавать.
       Z=(Z1,Z2,Z0) - комплексные сопротивление ветви (Ом) прямой, обратной и нулевой последовательностей
       E=(E1,E2,E0) - комплексные фазные значения Э.Д.С. (Вольт) прямой, обратной и нулевой последовательностей
       B=(B1,B2,B0) - комплексные значения поперечной емкостной проводимости B (См)
                       прямой, обратной и нулевой последовательностей,
                       если pl62w+ или аналогичная выдает Например
                       B1 = В2 = 90 мкСм (1/Ом*10^-6), B0 = 60 мкСм (1/Ом*10^-6)
                       то при создании ветви надо заполнять параметры ветви
                       B=(90e-6j,90e-6j,60e-6j)
       T=(Ktrans,GrT) - безразмерные параметры трансформаторной ветви:
          Ktrans - коэффициент трансформации силового трансформатора
          GrT - группа обмоток обмотки подключенной к узлу 2 (от 0 до 11)

    Результатом конструктора ветви является объект ветви, который используется для
    формирования расчетной модели и вывода результатов расчетов

    Изменить параметры ветви p можно с помощью метода
    p.edit(name,q1,q2,Z)
    p.edit(name,q1,q2,Z,desc='Примечание')
    p.edit(name,q1,q2,Z,E=(E1,E2,E0))
    p.edit(name,q1,q2,Z,B=(B1,B2,B0))
    p.edit(name,q1,q2,Z,T=(Ktrans,GrT))

    Пользовательские функции для объекта ветви p
    Вывод на экран параметров ветви - ее номера, названия, номеров и наименований узлов к которым она подключена,
    электрических параметров Z,E,B и T
    p.par()

    Вывод сводной таблицы результатов расчетов для ветви p
    со стороны 1-ого и 2-ого узла соответственно (направление токов и пр. в линию)
    p.res1()
    p.res2()

    Вывод конкретного параметра ParName в виде компексного числа
    для последующего использования в расчетах
    со стороны 1-ого и 2-ого узла соответственно (направление токов и пр. в линию)
    p.res1(ParName)
    p.res2(ParName)
    где ParName может принимать значения:
    'U1','U2','U0','3U0','UA','UB','UC','UABC','UAB','UBC','UCA','UAB_BC_CA',
    'I1','I2','I0','3I0','IA','IB','IC','IABC','IAB','IBC','ICA','IAB_BC_CA',
    'Z1','Z2','Z0','ZA','ZB','ZC','ZABC','ZAB','ZBC','ZCA','ZAB_BC_CA',
    'S1','S2','S0','SA','SB','SC','SABC','SAB','SBC','SCA','SAB_BC_CA','S'

    Вывод конкретного параметра ParName в заданной форме Form:
    p.res1(ParName,Form)
    p.res2(ParName,Form)
    где Form может принимать значения
    'R' - Активная составляющая
    'X' - Реактивная составляющая
    'M' - Модуль комплексного числа
    '<f' - Фаза вектора в градусах
    'R+jX' - Текстовый вид комплексного числа
    'M<f' - Текстовый вид комплексного числа


    Еще один способ получения конкректного параметра результата в виде
    компексного числа для его последующего использования в расчетах
    p.ParName
    где ParName может принимать значения:
    значения токов от 1-ого ко 2-ому узлу без учета емкостной проводимости
    I1,I2,I0,I120,IA,IB,IC,IABC,IAB,IBC,ICA,IAB_BC_CA
    со стороны 1-ого узла
    q1U1,q1U2,q1U0,q1U120,q1UA,q1UB,q1UC,q1UABC,q1UAB,q1UBC,q1UCA,q1UAB_BC_CA,
    q1I1,q1I2,q1I0,q1I120,q1IA,q1IB,q1IC,q1IABC,q1IAB,q1IBC,q1ICA,q1IAB_BC_CA,
    q1Z1,q1Z2,q1Z0,q1Z120,q1ZA,q1ZB,q1ZC,q1ZABC,q1ZAB,q1ZBC,q1ZCA,q1ZAB_BC_CA,
    q1S1,q1S2,q1S0,q1S120,q1SA,q1SB,q1SC,q1SABC,q1SAB,q1SBC,q1SCA,q1SAB_BC_CA,q1S
    со стороны 2-ого узла
    q2U1,q2U2,q2U0,q1U120,q2UA,q2UB,q2UC,q2UABC,q2UAB,q2UBC,q2UCA,q2UAB_BC_CA,
    q2I1,q2I2,q2I0,q1I120,q2IA,q2IB,q2IC,q2IABC,q2IAB,q2IBC,q2ICA,q2IAB_BC_CA,
    q2Z1,q2Z2,q2Z0,q1Z120,q2ZA,q2ZB,q2ZC,q2ZABC,q2ZAB,q2ZBC,q2ZCA,q2ZAB_BC_CA,
    q2S1,q2S2,q2S0,q1S120,q2SA,q2SB,q2SC,q2SABC,q2SAB,q2SBC,q2SCA,q2SAB_BC_CA,q2S
    '''
    def __init__(self,model,name,q1,q2,Z,E=(0, 0, 0),T=(1, 0),B=(0, 0, 0),desc=''):
        ''' Конструктор ветви
        P(model,name,q1,q2,Z) - простая ветвь
        P(model,name,q1,q2,Z,desc='Примечание') - ветвь с текстовым примечанием
        P(model,name,q1,q2,Z,E=(E1,E2,E0)) - ветвь представляющая энергосистему, генератор (Вольт - фазные)
        P(model,name,q1,q2,Z,B=(B1,B2,B0)) - ветвь c наличием поперечной емкостной проводимостью B/2 (См)
        P(model,name,q1,q2,Z,T=(Ktrans,GrT)) - ветвь представляющая трансформатор
        где:
           model - объект расчетной модели в которой создается ветвь
           name - краткое название ветви, обращение к ветви по ее имени не предусмотрено
           q1,q2 - число 0, что означает подключение ветви соответствующим концом к земле,
                   объект узла принадлежащего той же расчетной модели
           desc - Примечание или любая другая текстовая информация, можно не задавать.
           Z=(Z1,Z2,Z0) - комплексные сопротивление ветви (Ом) прямой, обратной и нулевой последовательностей
           E=(E1,E2,E0) - комплексные фазные значения Э.Д.С. (Вольт) прямой, обратной и нулевой последовательностей
           B=(B1,B2,B0) - комплексные значения поперечной емкостной проводимости B (См)
                           прямой, обратной и нулевой последовательностей,
                           если pl62w+ или аналогичная выдает Например
                           B1 = В2 = 90 мкСм (1/Ом*10^-6), B0 = 60 мкСм (1/Ом*10^-6)
                           то при создании ветви надо заполнять параметры ветви
                           B=(90e-6j,90e-6j,60e-6j)
           T=(Ktrans,GrT) - безразмерные параметры трансформаторной ветви:
              Ktrans - коэффициент трансформации силового трансформатора
              GrT - группа обмоток обмотки подключенной к узлу 2 (от 0 до 11)
        Результатом конструктора ветви является объект ветви, который используется для
        формирования расчетной модели и вывода результатов расчетов'''
        if not isinstance(model, Model):
            raise TypeError('Ошибка при добавлении ветви -', name, '\n',
                            'Аргумент model должен иметь тип Model!')
        if isinstance(q1, int):
            if q1 != 0:
                raise ValueError('Ошибка при добавлении ветви -', name, '\n',
                                 'Для подключения ветви к земле q1=0')
        elif isinstance(q1, Q):
            if not q1.model is model:
                raise ValueError('Ошибка при добавлении ветви -', name, '\n',
                                'Узел q1 должен принадлежать той-же модели!')
        else:
            raise TypeError('Ошибка при добавлении ветви -', name, '\n',
                            'Аргумент q1 должен иметь тип Q или int!')
        if isinstance(q2, int):
            if q2 != 0:
                raise ValueError('Ошибка при добавлении ветви -', name, '\n',
                                 'Для подключения ветви к земле q2=0')
        elif isinstance(q2, Q):
            if not q2.model is model:
                raise ValueError('Ошибка при добавлении ветви -', name, '\n',
                                'Узел q2 должен принадлежать той-же модели!')
        else:
            raise TypeError('Ошибка при добавлении ветви -', name, '\n',
                            'Аргумент q2 должен иметь тип Q или int!')
        if  q1 is q2:
            print('Предупреждение! при добавлении ветви -', name, '\n',
                            'Ветвь подключается обоими концами к одному и тому же узлу!')
        model.np+=1
        model.bp.append(self)
        self.id=model.np
        self.model=model
        self.name=name
        self.desc=desc
        self.q1=q1
        if isinstance(q1, Q):
            q1.addp(self)
        self.q2=q2
        if isinstance(q2, Q):
            q2.addp(self)
        self.Z=Z
        self.E=E
        self.T=T
        self.B=B
        self.mlist=[]
        self.kn=None

    def edit(self,name,q1,q2,Z,E=(0, 0, 0),T=(1, 0),B=(0, 0, 0),desc=''):
        '''
        Изменить параметры ветви можно с помощью метода
        p.edit(name,q1,q2,Z)
        p.edit(name,q1,q2,Z,desc='Примечание')
        p.edit(name,q1,q2,Z,E=(E1,E2,E0))
        p.edit(name,q1,q2,Z,B=(B1,B2,B0))
        p.edit(name,q1,q2,Z,T=(Ktrans,GrT))
        '''
        if isinstance(q1, int):
            if q1 != 0:
                raise ValueError('Ошибка при редактировании ветви №', self.id, ' - ', self.name, '\n',
                                 'Для подключения ветви к земле q1=0')
        elif isinstance(q1, Q):
            if not q1.model is self.model:
                raise ValueError('Ошибка при редактировании ветви №', self.id, ' - ', self.name, '\n',
                                'Узел q1 должен принадлежать той-же модели!')
        else:
            raise TypeError('Ошибка при редактировании ветви №', self.id, ' - ', self.name, '\n',
                            'Аргумент q1 должен иметь тип Q или int!')
        if isinstance(q2, int):
            if q2 != 0:
                raise ValueError('Ошибка при редактировании ветви №', self.id, ' - ', self.name, '\n',
                                 'Для подключения ветви к земле q2=0')
        elif isinstance(q2, Q):
            if not q2.model is self.model:
                raise ValueError('Ошибка при редактировании ветви №', self.id, ' - ', self.name, '\n',
                                'Узел q2 должен принадлежать той-же модели!')
        else:
            raise TypeError('Ошибка при редактировании ветви №', self.id, ' - ', self.name, '\n',
                            'Аргумент q2 должен иметь тип Q или int!')
        if  q1 is q2:
            print('Предупреждение! при добавлении ветви -', name, '\n',
                            'Ветвь подключается обоими концами к одному и тому же узлу!')
        self.name=name
        self.desc=desc
        self.q1=q1
        if isinstance(q1, Q):
            q1.addp(self)
            q1.update()
        self.q2=q2
        if isinstance(q2, Q):
            q2.addp(self)
            q2.update()
        self.Z=Z
        self.E=E
        self.T=T
        self.B=B

    def addm(self,mid):
        '''Служебный метод, предназачен для информирования ветви
        о подключенных к ней взаимоиндуктивностей'''
        self.mlist.append(mid)

    def setn(self,kn):
        '''Служебный метод, предназачен для информирования ветви
        о наличии на ней обрыва'''
        self.kn=kn

    def par(self):
        '''Вывод на экран параметров ветви - ее номера, названия, номеров и наименований узлов к которым она подключена,
        электрических параметров Z,E,B и T
        p.par()'''
        if isinstance(self.q1, Q):
            q1id = self.q1.id; q1name = self.q1.name
        else:
            q1id = 0; q1name = 'Земля'
        if isinstance(self.q2, Q):
            q2id = self.q2.id; q2name = self.q2.name
        else:
            q2id = 0; q2name = 'Земля'
        print('Ветвь № {} - {} : {}({}) <=> {}({})'.format(self.id,self.name,q1id,q1name,q2id,q2name))
        print('Z = {}; E = {}; T = {}; B = {}'.format(self.Z,self.E,self.T,self.B))

    def getres(self):
        '''Служебный метод, возвращает результат расчета по данной ветви
        без учета наличия поперечной проводимости и направления от узла 1 к узлу 2
        токов прямой, обратной и нулевой последовательностей, тоже что и p.res1('U120') если B=0'''
        if self.model is None:
            raise ValueError('Ошибка при выводе результатов расчетов Ветви №', self.id, ' - ', self.name, '\n',
                            'Ветвь не принадлежит какой либо модели!')
        if self.model.X is None:
            raise ValueError('Ошибка при выводе результатов расчетов Ветви №', self.id, ' - ', self.name, '\n',
                            'Не произведен расчет электрических величин!')
        pId = 3*(self.id-1)
        return self.model.X[pId:pId+3]

    def getresq1(self,i1,i2,i0):
        '''Служебный метод, возвращает результат расчета по данной ветви
        c учетом наличия поперечной проводимости и направления от узла 1 к узлу 2
        токов прямой, обратной и нулевой последовательностей, тоже что и p.res1('U120')'''
        if isinstance(self.q1, Q):
            u1,u2,u0 = self.q1.getres()
        else:
            u1,u2,u0 = [0j,0j,0j]
        i1 += u1 * self.B[0]/2
        i2 += u2 * self.B[1]/2
        i0 += u0 * self.B[2]/2
        return [u1,u2,u0,i1,i2,i0]

    def getresq2(self,i1,i2,i0):
        '''Служебный метод, возвращает результат расчета по данной ветви
        c учетом наличия поперечной проводимости и направления от узла 2 к узлу 1
        токов прямой, обратной и нулевой последовательностей, тоже что и p.res2('U120')'''
        if isinstance(self.q2, Q):
            u1,u2,u0 = self.q2.getres()
        else:
            u1,u2,u0 = [0j,0j,0j]
        Kt = self.T[0]
        GrT = self.T[1]
        Kt1=Kt*np.exp(Kf*GrT)
        if GrT % 2==0:
            Kt2=Kt1
        else:
            Kt2=np.conj(Kt1)
        Kt0=Kt1
        i1 = -Kt1*i1 + u1 * self.B[0]/2
        i2 = -Kt2*i2 + u2 * self.B[1]/2
        i0 = -Kt0*i0 + u0 * self.B[2]/2
        return [u1,u2,u0,i1,i2,i0]

    def res1(self,parname='',subpar=''):
        '''Вывод сводной таблицы результатов расчетов для ветви p
        со стороны 1-ого узла (направление токов и пр. в линию)
        p.res1()

        Вывод конкретного параметра ParName в виде компексного числа
        для последующего использования в расчетах
        со стороны 1-ого узла (направление токов и пр. в линию)
        p.res1(ParName)
        где ParName может принимать значения:
        'U1','U2','U0','3U0','UA','UB','UC','UABC','UAB','UBC','UCA','UAB_BC_CA',
        'I1','I2','I0','3I0','IA','IB','IC','IABC','IAB','IBC','ICA','IAB_BC_CA',
        'Z1','Z2','Z0','ZA','ZB','ZC','ZABC','ZAB','ZBC','ZCA','ZAB_BC_CA',
        'S1','S2','S0','SA','SB','SC','SABC','SAB','SBC','SCA','SAB_BC_CA','S'

        Вывод конкретного параметра ParName в заданной форме Form:
        p.res1(ParName,Form)
        где Form может принимать значения
        'R' - Активная составляющая
        'X' - Реактивная составляющая
        'M' - Модуль комплексного числа
        '<f' - Фаза вектора в градусах
        'R+jX' - Текстовый вид комплексного числа
        'M<f' - Текстовый вид комплексного числа '''
        i1,i2,i0 = self.getres()
        if isinstance(self.q1, Q):
            q1id = self.q1.id
            q1name = self.q1.name
        else:
            q1id = 0
            q1name = 'Земля'
        u1,u2,u0,i1,i2,i0 = self.getresq1(i1,i2,i0)
        iA,iB,iC,iAB,iBC,iCA = msymm2faze(i1,i2,i0)
        uA,uB,uC,uAB,uBC,uCA = msymm2faze(u1,u2,u0)
        if parname=='':
            print("Ветвь № {} - {}".format(self.id, self.name))
            print("Значения токов по ветви со стороны узла №{} - {}\n".format(q1id, q1name))
            print("I1  = {0:>7.0f} < {1:>6.1f} | I2  = {2:>7.0f} < {3:>6.1f} | 3I0 = {4:>7.0f} < {5:>6.1f}".format(np.abs(i1),r2d*np.angle(i1),np.abs(i2),r2d*np.angle(i2),np.abs(3*i0),r2d*np.angle(i0)))
            print("IA  = {0:>7.0f} < {1:>6.1f} | IB  = {2:>7.0f} < {3:>6.1f} | IC  = {4:>7.0f} < {5:>6.1f}".format(np.abs(iA),r2d*np.angle(iA),np.abs(iB),r2d*np.angle(iB),np.abs(iC),r2d*np.angle(iC)))
            print("IAB = {0:>7.0f} < {1:>6.1f} | IBC = {2:>7.0f} < {3:>6.1f} | ICA = {4:>7.0f} < {5:>6.1f}".format(np.abs(iAB),r2d*np.angle(iAB),np.abs(iBC),r2d*np.angle(iBC),np.abs(iCA),r2d*np.angle(iCA)))
            print("Значения напряжения в узле №{} - {}\n".format(q1id, q1name))
            print("U1  = {0:>7.0f} < {1:>6.1f} | U2  = {2:>7.0f} < {3:>6.1f} | 3U0 = {4:>7.0f} < {5:>6.1f}".format(np.abs(u1),r2d*np.angle(u1),np.abs(u2),r2d*np.angle(u2),np.abs(3*u0),r2d*np.angle(u0)))
            print("UA  = {0:>7.0f} < {1:>6.1f} | UB  = {2:>7.0f} < {3:>6.1f} | UC  = {4:>7.0f} < {5:>6.1f}".format(np.abs(uA),r2d*np.angle(uA),np.abs(uB),r2d*np.angle(uB),np.abs(uC),r2d*np.angle(uC)))
            print("UAB = {0:>7.0f} < {1:>6.1f} | UBC = {2:>7.0f} < {3:>6.1f} | UCA = {4:>7.0f} < {5:>6.1f}".format(np.abs(uAB),r2d*np.angle(uAB),np.abs(uBC),r2d*np.angle(uBC),np.abs(uCA),r2d*np.angle(uCA)))
        else:
            res = mselectz[parname]([u1,u2,u0],[i1,i2,i0])
            if isinstance(res, (tuple, list)):
                return mform3[subpar](res,parname)
            else:
                return mform1[subpar](res,parname)

    def res2(self,parname='',subpar=''):
        '''Вывод сводной таблицы результатов расчетов для ветви p
        со стороны 2-ого узла (направление токов и пр. в линию)
        p.res1()

        Вывод конкретного параметра ParName в виде компексного числа
        для последующего использования в расчетах
        со стороны 2-ого узла (направление токов и пр. в линию)
        p.res2(ParName)
        где ParName может принимать значения:
        'U1','U2','U0','3U0','U120','UA','UB','UC','UABC','UAB','UBC','UCA','UAB_BC_CA',
        'I1','I2','I0','3I0','I120','IA','IB','IC','IABC','IAB','IBC','ICA','IAB_BC_CA',
        'Z1','Z2','Z0','Z120','ZA','ZB','ZC','ZABC','ZAB','ZBC','ZCA','ZAB_BC_CA',
        'S1','S2','S0','S120','SA','SB','SC','SABC','SAB','SBC','SCA','SAB_BC_CA','S'

        Вывод конкретного параметра ParName в заданной форме Form:
        p.res2(ParName,Form)
        где Form может принимать значения
        'R' - Активная составляющая
        'X' - Реактивная составляющая
        'M' - Модуль комплексного числа
        '<f' - Фаза вектора в градусах
        'R+jX' - Текстовый вид комплексного числа
        'M<f' - Текстовый вид комплексного числа '''
        i1,i2,i0 = self.getres()
        if isinstance(self.q2, Q):
            q2id = self.q2.id
            q2name = self.q2.name
        else:
            q2id = 0
            q2name = 'Земля'
        u1,u2,u0,i1,i2,i0 = self.getresq2(i1,i2,i0)
        iA,iB,iC,iAB,iBC,iCA = msymm2faze(i1,i2,i0)
        uA,uB,uC,uAB,uBC,uCA = msymm2faze(u1,u2,u0)
        if parname=='':
            print("Ветвь № {} - {}".format(self.id, self.name))
            print("Значения токов по ветви со стороны узла №{} - {}\n".format(q2id, q2name))
            print("I1  = {0:>7.0f} < {1:>6.1f} | I2  = {2:>7.0f} < {3:>6.1f} | 3I0 = {4:>7.0f} < {5:>6.1f}".format(np.abs(i1),r2d*np.angle(i1),np.abs(i2),r2d*np.angle(i2),np.abs(3*i0),r2d*np.angle(i0)))
            print("IA  = {0:>7.0f} < {1:>6.1f} | IB  = {2:>7.0f} < {3:>6.1f} | IC  = {4:>7.0f} < {5:>6.1f}".format(np.abs(iA),r2d*np.angle(iA),np.abs(iB),r2d*np.angle(iB),np.abs(iC),r2d*np.angle(iC)))
            print("IAB = {0:>7.0f} < {1:>6.1f} | IBC = {2:>7.0f} < {3:>6.1f} | ICA = {4:>7.0f} < {5:>6.1f}".format(np.abs(iAB),r2d*np.angle(iAB),np.abs(iBC),r2d*np.angle(iBC),np.abs(iCA),r2d*np.angle(iCA)))
            print("Значения напряжения в узле №{} - {}\n".format(q2id, q2name))
            print("U1  = {0:>7.0f} < {1:>6.1f} | U2  = {2:>7.0f} < {3:>6.1f} | 3U0 = {4:>7.0f} < {5:>6.1f}".format(np.abs(u1),r2d*np.angle(u1),np.abs(u2),r2d*np.angle(u2),np.abs(3*u0),r2d*np.angle(u0)))
            print("UA  = {0:>7.0f} < {1:>6.1f} | UB  = {2:>7.0f} < {3:>6.1f} | UC  = {4:>7.0f} < {5:>6.1f}".format(np.abs(uA),r2d*np.angle(uA),np.abs(uB),r2d*np.angle(uB),np.abs(uC),r2d*np.angle(uC)))
            print("UAB = {0:>7.0f} < {1:>6.1f} | UBC = {2:>7.0f} < {3:>6.1f} | UCA = {4:>7.0f} < {5:>6.1f}".format(np.abs(uAB),r2d*np.angle(uAB),np.abs(uBC),r2d*np.angle(uBC),np.abs(uCA),r2d*np.angle(uCA)))
        else:
            res = mselectz[parname]([u1,u2,u0],[i1,i2,i0])
            if isinstance(res, (tuple, list)):
                return mform3[subpar](res,parname)
            else:
                return mform1[subpar](res,parname)

    def __repr__(self):
        '''
        Еще один способ вывода сводной таблицы результатов расчетов для ветви p
        В командной строке интерпретара набрать название переменной объекта ветви и нажать Enter
        p Enter, выводятся результаты с обоих концов ветви
        '''
        pi1,pi2,pi0 = self.getres()
        if isinstance(self.q1, Q):
            q1id = self.q1.id
            q1name = self.q1.name
        else:
            q1id = 0
            q1name = 'Земля'
        u1,u2,u0,i1,i2,i0 = self.getresq1(pi1,pi2,pi0)
        iA,iB,iC,iAB,iBC,iCA = msymm2faze(i1,i2,i0)
        uA,uB,uC,uAB,uBC,uCA = msymm2faze(u1,u2,u0)
        strres = ("Ветвь № {} - {}\n".format(self.id, self.name))
        strres += ("Значения токов по ветви со стороны узла №{} - {}\n".format(q1id, q1name))
        strres += ("I1  = {0:>7.0f} < {1:>6.1f} | I2  = {2:>7.0f} < {3:>6.1f} | 3I0 = {4:>7.0f} < {5:>6.1f}\n".format(np.abs(i1),r2d*np.angle(i1),np.abs(i2),r2d*np.angle(i2),np.abs(3*i0),r2d*np.angle(i0)))
        strres += ("IA  = {0:>7.0f} < {1:>6.1f} | IB  = {2:>7.0f} < {3:>6.1f} | IC  = {4:>7.0f} < {5:>6.1f}\n".format(np.abs(iA),r2d*np.angle(iA),np.abs(iB),r2d*np.angle(iB),np.abs(iC),r2d*np.angle(iC)))
        strres += ("IAB = {0:>7.0f} < {1:>6.1f} | IBC = {2:>7.0f} < {3:>6.1f} | ICA = {4:>7.0f} < {5:>6.1f}\n".format(np.abs(iAB),r2d*np.angle(iAB),np.abs(iBC),r2d*np.angle(iBC),np.abs(iCA),r2d*np.angle(iCA)))
        strres += ("Значения напряжения в узле №{} - {}\n".format(q1id, q1name))
        strres += ("U1  = {0:>7.0f} < {1:>6.1f} | U2  = {2:>7.0f} < {3:>6.1f} | 3U0 = {4:>7.0f} < {5:>6.1f}\n".format(np.abs(u1),r2d*np.angle(u1),np.abs(u2),r2d*np.angle(u2),np.abs(3*u0),r2d*np.angle(u0)))
        strres += ("UA  = {0:>7.0f} < {1:>6.1f} | UB  = {2:>7.0f} < {3:>6.1f} | UC  = {4:>7.0f} < {5:>6.1f}\n".format(np.abs(uA),r2d*np.angle(uA),np.abs(uB),r2d*np.angle(uB),np.abs(uC),r2d*np.angle(uC)))
        strres += ("UAB = {0:>7.0f} < {1:>6.1f} | UBC = {2:>7.0f} < {3:>6.1f} | UCA = {4:>7.0f} < {5:>6.1f}\n".format(np.abs(uAB),r2d*np.angle(uAB),np.abs(uBC),r2d*np.angle(uBC),np.abs(uCA),r2d*np.angle(uCA)))
        if isinstance(self.q2, Q):
            q2id = self.q2.id
            q2name = self.q2.name
        else:
            q2id = 0
            q2name = 'Земля'
        u1,u2,u0,i1,i2,i0 = self.getresq2(pi1,pi2,pi0)
        iA,iB,iC,iAB,iBC,iCA = msymm2faze(i1,i2,i0)
        uA,uB,uC,uAB,uBC,uCA = msymm2faze(u1,u2,u0)
        strres += ("Значения токов по ветви со стороны узла №{} - {}\n".format(q2id, q2name))
        strres += ("I1  = {0:>7.0f} < {1:>6.1f} | I2  = {2:>7.0f} < {3:>6.1f} | 3I0 = {4:>7.0f} < {5:>6.1f}\n".format(np.abs(i1),r2d*np.angle(i1),np.abs(i2),r2d*np.angle(i2),np.abs(3*i0),r2d*np.angle(i0)))
        strres += ("IA  = {0:>7.0f} < {1:>6.1f} | IB  = {2:>7.0f} < {3:>6.1f} | IC  = {4:>7.0f} < {5:>6.1f}\n".format(np.abs(iA),r2d*np.angle(iA),np.abs(iB),r2d*np.angle(iB),np.abs(iC),r2d*np.angle(iC)))
        strres += ("IAB = {0:>7.0f} < {1:>6.1f} | IBC = {2:>7.0f} < {3:>6.1f} | ICA = {4:>7.0f} < {5:>6.1f}\n".format(np.abs(iAB),r2d*np.angle(iAB),np.abs(iBC),r2d*np.angle(iBC),np.abs(iCA),r2d*np.angle(iCA)))
        strres += ("Значения напряжения в узле №{} - {}\n".format(q2id, q2name))
        strres += ("U1  = {0:>7.0f} < {1:>6.1f} | U2  = {2:>7.0f} < {3:>6.1f} | 3U0 = {4:>7.0f} < {5:>6.1f}\n".format(np.abs(u1),r2d*np.angle(u1),np.abs(u2),r2d*np.angle(u2),np.abs(3*u0),r2d*np.angle(u0)))
        strres += ("UA  = {0:>7.0f} < {1:>6.1f} | UB  = {2:>7.0f} < {3:>6.1f} | UC  = {4:>7.0f} < {5:>6.1f}\n".format(np.abs(uA),r2d*np.angle(uA),np.abs(uB),r2d*np.angle(uB),np.abs(uC),r2d*np.angle(uC)))
        strres += ("UAB = {0:>7.0f} < {1:>6.1f} | UBC = {2:>7.0f} < {3:>6.1f} | UCA = {4:>7.0f} < {5:>6.1f}\n".format(np.abs(uAB),r2d*np.angle(uAB),np.abs(uBC),r2d*np.angle(uBC),np.abs(uCA),r2d*np.angle(uCA)))
        return (strres)

    def __getattr__(self, attrname):
        '''
        Еще один способ получения конкректного параметра результата в виде
        компексного числа для его последующего использования в расчетах
        p.ParName
        где ParName может принимать значения:
        значения токов от 1-ого ко 2-ому узлу без учета емкостной проводимости
        I1,I2,I0,I120,IA,IB,IC,IABC,IAB,IBC,ICA,IAB_BC_CA
        со стороны 1-ого узла
        q1U1,q1U2,q1U0,q1U120,q1UA,q1UB,q1UC,q1UABC,q1UAB,q1UBC,q1UCA,q1UAB_BC_CA,
        q1I1,q1I2,q1I0,q1I120,q1IA,q1IB,q1IC,q1IABC,q1IAB,q1IBC,q1ICA,q1IAB_BC_CA,
        q1Z1,q1Z2,q1Z0,q1Z120,q1ZA,q1ZB,q1ZC,q1ZABC,q1ZAB,q1ZBC,q1ZCA,q1ZAB_BC_CA,
        q1S1,q1S2,q1S0,q1S120,q1SA,q1SB,q1SC,q1SABC,q1SAB,q1SBC,q1SCA,q1SAB_BC_CA,q1S
        со стороны 2-ого узла
        q2U1,q2U2,q2U0,q1U120,q2UA,q2UB,q2UC,q2UABC,q2UAB,q2UBC,q2UCA,q2UAB_BC_CA,
        q2I1,q2I2,q2I0,q1I120,q2IA,q2IB,q2IC,q2IABC,q2IAB,q2IBC,q2ICA,q2IAB_BC_CA,
        q2Z1,q2Z2,q2Z0,q1Z120,q2ZA,q2ZB,q2ZC,q2ZABC,q2ZAB,q2ZBC,q2ZCA,q2ZAB_BC_CA,
        q2S1,q2S2,q2S0,q1S120,q2SA,q2SB,q2SC,q2SABC,q2SAB,q2SBC,q2SCA,q2SAB_BC_CA,q2S
        '''
        pi1,pi2,pi0 = self.getres()
        if not attrname[:2] in ('q1', 'q2'):
            res = mselectz[attrname]([0j,0j,0j],[pi1,pi2,pi0])
        elif attrname[:2] == 'q1':
            u1,u2,u0,i1,i2,i0 = self.getresq1(pi1,pi2,pi0)
            res = mselectz[attrname[2:]]([u1,u2,u0],[i1,i2,i0])
        elif attrname[:2] == 'q2':
            u1,u2,u0,i1,i2,i0 = self.getresq2(pi1,pi2,pi0)
            res = mselectz[attrname[2:]]([u1,u2,u0],[i1,i2,i0])
        return res

class M:
    '''
    Класс взаимоиндукции нулевой последовательности,
    необходим для формирования расчетной модели

    Создание ветви с помощью конструктора
    M(model,name,p1,p2,M12,M21) - взаимоиндукция
    M(model,name,p1,p2,M12,M21,desc='Примечание') - взаимоиндукция с текстовым примечанием
    где:
       model - объект расчетной модели в которой создается взаимоиндукция
       name - краткое название взаимоиндукции, обращение к ветви по ее имени не предусмотрено
       p1,p2 - объекты ветви принадлежащего той же расчетной модели между которыми создается взаимоиндукция
       desc - Примечание или любая другая текстовая информация, можно не задавать.
       M12 - взаимоиндукция влияния ветви p2 на ветвь p1
       M21 - взаимоиндукция влияния ветви p1 на ветвь p2

    Результатом конструктора ветви является объект взаимоиндукции, который используется для
    формирования расчетной модели

    Изменить параметры взаимоиндукции m можно с помощью метода
    m.edit(name,M12,M21)

    Пользовательские функции для объекта взаимоиндукции m
    Вывод на экран параметров ветви - ее номера, названия, номеров и наименований ветвей
    между которыми создана взаимоиндукция, электрических параметров M12,M21
    m.par()
    '''
    def __init__(self,model,name,p1,p2,M12,M21,desc=''):
        ''' Конструктор взаимоиндукции
        Создание ветви с помощью конструктора
        M(model,name,p1,p2,M12,M21) - взаимоиндукция
        M(model,name,p1,p2,M12,M21,desc='Примечание') - взаимоиндукция с текстовым примечанием
        где:
           model - объект расчетной модели в которой создается взаимоиндукция
           name - краткое название взаимоиндукции, обращение к ветви по ее имени не предусмотрено
           p1,p2 - объекты ветви принадлежащего той же расчетной модели между которыми создается взаимоиндукция
           desc - Примечание или любая другая текстовая информация, можно не задавать.
           M12 - взаимоиндукция влияния ветви p2 на ветвь p1
           M21 - взаимоиндукция влияния ветви p1 на ветвь p2
        '''
        if not isinstance(model, Model):
            raise TypeError('Ошибка при добавлении взаимоиндукции -', name, '\n',
                            'Аргумент model должен иметь тип Model!')
        if not isinstance(p1, P):
            raise TypeError('Ошибка при добавлении взаимоиндукции -', name, '\n',
                            'Аргумент p1 должен иметь тип P!')
        if not isinstance(p2, P):
            raise TypeError('Ошибка при добавлении взаимоиндукции -', name, '\n',
                            'Аргумент p2 должен иметь тип P!')
        if not p1.model is model:
            raise ValueError('Ошибка при добавлении взаимоиндукции -', name, '\n',
                            'Ветвь p1 должна принадлежать той-же модели!')
        if not p2.model is model:
            raise ValueError('Ошибка при добавлении взаимоиндукции -', name, '\n',
                            'Ветвь p2 должна принадлежать той-же модели!')
        if  p1 is p2:
            raise ValueError('Ошибка при добавлении взаимоиндукции -', name, '\n',
                            'Взаимоиндукция подключается к одной и той же ветви!')
        model.nm+=1
        model.bm.append(self)
        self.id=model.nm
        self.model=model
        self.name=name
        self.desc=desc
        self.p1=p1
        p1.addm(self)
        self.p2=p2
        p2.addm(self)
        self.M12=M12
        self.M21=M21

    def edit(self,name,M12,M21):
        ''' Редактирование взаимоиндукции
        m.edit(model,name,M12,M21)'''
        self.name=name
        self.M12=M12
        self.M21=M21

    def par(self):
        '''Вывод на экран параметров ветви - ее номера, названия, номеров и наименований ветвей
        между которыми создана взаимоиндукция, электрических параметров M12,M21
        m.par()
        '''
        print('Взаимоиндукция № {} - {} : {}({}) <=> {}({})'.format(self.id,self.name,self.p1.id,self.p1.name,self.p2.id,self.p2.name))
        print('M12 = {}; M21 = {}'.format(self.M12,self.M21))

class N:
    '''
    Класс продольной (обрыв) или поперечной (КЗ) несимметрии,
    необходим для формирования расчетной модели и получения результатов расчета

    Создание несимметрии с помощью конструктора
    N(model,name,qp,SC) - несимметрия
    N(model,name,qp,SC,desc='Примечание') - несимметрия с текстовым примечанием
    N(model,name,qp,SC,r=Rd) - несимметрия в виде КЗ с переходным сопротивлением
    где:
       model - объект расчетной модели в которой создается несимметрия
       name - краткое название несимметрии, обращение к несимметрии по ее имени не предусмотрено
       qp - объект узла (КЗ) или ветви (обрыв) в котором создается несимметрия
       desc - Примечание или любая другая текстовая информация, можно не задавать.
       SC - вид КЗ, обрыва может принимать значения:
           'A0','B0','C0' - металлические однофазные КЗ на землю или обрыв соответствующей фазы
           'A0r','B0r','C0r' - однофазные КЗ на землю через переходное сопротивление
           'AB','BC','CA' - металлические двухфазные КЗ  или обрыв соответствующих фаз
           'ABr','BCr','CAr' - двухфазные КЗ через переходное сопротивление
           'AB0','BC0','CA0' - металлические двухфазные КЗ на землю
           'ABC' - трехфазное КЗ без земли  или обрыв трех фаз
           'ABC0' - трехфазное КЗ на землю
           'N0' - Заземление в узле в схеме нулевой последовательности
                  или обрыв по нулевой последовательности на ветви

    Результатом конструктора несимметрии является объект несимметрии, который используется для
    формирования расчетной модели и вывода результатов расчетов

    Изменить параметры несимметрии n можно с помощью метода
    n.edit(name,SC)
    n.edit(name,SC,desc='')
    n.edit(name,SC,r=0)

    Пользовательские функции для объекта несимметрии n
    Вывод на экран параметров несимметрии - ее номера, названия,
    номера и наименования узла или ветви к которым она подключена,
    вида несимметрии
    n.par()

    Вывод сводной таблицы результатов расчетов для несимметрии n
    n.res()

    Вывод конкретного параметра ParName в виде компексного числа
    для последующего использования в расчетах
    n.res(ParName)
    где ParName может принимать значения:
    'U1','U2','U0','3U0','U120','UA','UB','UC','UABC','UAB','UBC','UCA','UAB_BC_CA',
    'I1','I2','I0','3I0','I120','IA','IB','IC','IABC','IAB','IBC','ICA','IAB_BC_CA',
    'Z1','Z2','Z0','Z120','ZA','ZB','ZC','ZABC','ZAB','ZBC','ZCA','ZAB_BC_CA',
    'S1','S2','S0','S120','SA','SB','SC','SABC','SAB','SBC','SCA','SAB_BC_CA','S'

    Вывод конкретного параметра ParName в заданной форме Form:
    n.res(ParName,Form)
    где Form может принимать значения
    'R' - Активная составляющая
    'X' - Реактивная составляющая
    'M' - Модуль комплексного числа
    '<f' - Фаза вектора в градусах
    'R+jX' - Текстовый вид комплексного числа
    'M<f' - Текстовый вид комплексного числа


    Еще один способ получения конкректного параметра результата в виде
    компексного числа для его последующего использования в расчетах
    n.ParName
    где ParName может принимать значения:
    U1,U2,U0,UA,UB,UC,UABC,UAB,UBC,UCA,UAB_BC_CA
    I1,I2,I0,IA,IB,IC,IABC,IAB,IBC,ICA,IAB_BC_CA
    Z1,Z2,Z0,Z120,ZA,ZB,ZC,ZABC,ZAB,ZBC,ZCA,ZAB_BC_CA,
    S1,S2,S0,S120,SA,SB,SC,SABC,SAB,SBC,SCA,SAB_BC_CA,S
    '''
    def __init__(self,model,name,qp,SC,r=0,desc=''):
        ''' Конструктор повреждения (КЗ или обрыва)'''
        if not isinstance(model, Model):
            raise TypeError('Ошибка при добавлении несимметрии -', name, '\n',
                            'Аргумент model должен иметь тип Model!')
        if not isinstance(qp, (Q,P)):
            raise TypeError('Ошибка при добавлении несимметрии -', name, '\n',
                            'Аргумент qp должен иметь тип Q или P!')
        if not qp.model is model:
            raise ValueError('Ошибка при добавлении несимметрии -', name, '\n',
                            'Узел/Ветвь qp должны принадлежать той-же модели!')
        model.nn+=1
        model.bn.append(self)
        self.id=model.nn
        self.model=model
        self.name=name
        self.desc=desc
        self.qp=qp
        qp.setn(self)
        self.SC=SC
        self.r=r

    def edit(self, name,SC,r=0,desc=''):
        '''
        Изменить параметры несимметрии n можно с помощью метода
        n.edit(name,SC)
        n.edit(name,SC,desc='')
        n.edit(name,SC,r=0)'''
        self.name=name
        self.desc=desc
        self.SC=SC
        self.r=r

    def par(self):
        '''
        Вывод на экран параметров несимметрии - ее номера, названия,
        номера и наименования узла или ветви к которым она подключена,
        вида несимметрии
        n.par()
        '''
        if isinstance(self.qp, Q):
            print('КЗ № {} - {} : {} (r={}) в узле № {}({})'.format(self.id,self.name,self.SC,self.r,self.qp.id,self.qp.name))
        elif isinstance(self.qp, P):
            print('Обрыв № {} - {} : {} на ветви № {}({})'.format(self.id,self.name,self.SC,self.qp.id,self.qp.name))

    def getres(self):
        '''Служебный метод, возвращает результат расчета по данной несимметрии
        для КЗ - токи КЗ прямой, обратной и нулевой последовательностей;
        для обрывов - напряжения продольной несимметрии прямой, обратной и нулевой последовательностей.
        '''
        if self.model is None:
            raise ValueError('Ошибка при выводе результатов расчетов несимметрии №', self.id, ' - ', self.name, '\n',
                            'Несимметрия не принадлежит какой либо модели!')
        if self.model.X is None:
            raise ValueError('Ошибка при выводе результатов расчетов несимметрии №', self.id, ' - ', self.name, '\n',
                            'Не произведен расчет электрических величин!')
        nId = 3*(self.model.np+self.model.nq+self.id-1)
        return self.model.X[nId:nId+3]

    def res(self,parname='',subpar=''):
        '''
        Вывод сводной таблицы результатов расчетов для несимметрии n
        n.res()

        Вывод конкретного параметра ParName в виде компексного числа
        для последующего использования в расчетах
        n.res(ParName)
        где ParName может принимать значения:
        'U1','U2','U0','3U0','U120','UA','UB','UC','UABC','UAB','UBC','UCA','UAB_BC_CA',
        'I1','I2','I0','3I0','I120','IA','IB','IC','IABC','IAB','IBC','ICA','IAB_BC_CA',
        'Z1','Z2','Z0','Z120','ZA','ZB','ZC','ZABC','ZAB','ZBC','ZCA','ZAB_BC_CA',
        'S1','S2','S0','S120','SA','SB','SC','SABC','SAB','SBC','SCA','SAB_BC_CA','S'

        Вывод конкретного параметра ParName в заданной форме Form:
        n.res(ParName,Form)
        где Form может принимать значения
        'R' - Активная составляющая
        'X' - Реактивная составляющая
        'M' - Модуль комплексного числа
        '<f' - Фаза вектора в градусах
        'R+jX' - Текстовый вид комплексного числа
        'M<f' - Текстовый вид комплексного числа
        '''
        if isinstance(self.qp, Q):
            u1,u2,u0 = self.qp.getres()
            i1,i2,i0 = self.getres()
            iA,iB,iC,iAB,iBC,iCA = msymm2faze(i1,i2,i0)
            uA,uB,uC,uAB,uBC,uCA = msymm2faze(u1,u2,u0)
            if parname=='':
                print('КЗ № {} - {} - {}'.format(self.id, self.name, self.SC))
                print('В Узле № {} - {}'.format(self.qp.id, self.qp.name))
                print("U1  = {0:>7.0f} < {1:>6.1f} | U2  = {2:>7.0f} < {3:>6.1f} | 3U0 = {4:>7.0f} < {5:>6.1f}".format(np.abs(u1),r2d*np.angle(u1),np.abs(u2),r2d*np.angle(u2),np.abs(3*u0),r2d*np.angle(u0)))
                print("UA  = {0:>7.0f} < {1:>6.1f} | UB  = {2:>7.0f} < {3:>6.1f} | UC  = {4:>7.0f} < {5:>6.1f}".format(np.abs(uA),r2d*np.angle(uA),np.abs(uB),r2d*np.angle(uB),np.abs(uC),r2d*np.angle(uC)))
                print("UAB = {0:>7.0f} < {1:>6.1f} | UBC = {2:>7.0f} < {3:>6.1f} | UCA = {4:>7.0f} < {5:>6.1f}".format(np.abs(uAB),r2d*np.angle(uAB),np.abs(uBC),r2d*np.angle(uBC),np.abs(uCA),r2d*np.angle(uCA)))
                print('Суммарный ток КЗ в Узле № {} - {}'.format(self.qp.id, self.qp.name))
                print("I1  = {0:>7.0f} < {1:>6.1f} | I2  = {2:>7.0f} < {3:>6.1f} | 3I0 = {4:>7.0f} < {5:>6.1f}".format(np.abs(i1),r2d*np.angle(i1),np.abs(i2),r2d*np.angle(i2),np.abs(3*i0),r2d*np.angle(i0)))
                print("IA  = {0:>7.0f} < {1:>6.1f} | IB  = {2:>7.0f} < {3:>6.1f} | IC  = {4:>7.0f} < {5:>6.1f}".format(np.abs(iA),r2d*np.angle(iA),np.abs(iB),r2d*np.angle(iB),np.abs(iC),r2d*np.angle(iC)))
                print("IAB = {0:>7.0f} < {1:>6.1f} | IBC = {2:>7.0f} < {3:>6.1f} | ICA = {4:>7.0f} < {5:>6.1f}".format(np.abs(iAB),r2d*np.angle(iAB),np.abs(iBC),r2d*np.angle(iBC),np.abs(iCA),r2d*np.angle(iCA)))
                print('Подтекание токов по ветвям')

                for kp in self.qp.plist:
                    i1,i2,i0 = kp.getres()
                    if self.qp is kp.q1:
                        u1,u2,u0,i1,i2,i0 = kp.getresq1(i1,i2,i0)
                    elif self.qp is kp.q2:
                        u1,u2,u0,i1,i2,i0 = kp.getresq2(i1,i2,i0)
                    i1 = -i1
                    i2 = -i2
                    i0 = -i0
                    iA = i1 + i2 + i0
                    iB = a2*i1 + a*i2 + i0
                    iC = a*i1 + a2*i2 + i0
                    print('Ветвь № {} - {}'.format(kp.id, kp.name))
                    print("I1  = {0:>7.0f} < {1:>6.1f} | I2  = {2:>7.0f} < {3:>6.1f} | 3I0 = {4:>7.0f} < {5:>6.1f}".format(np.abs(i1),r2d*np.angle(i1),np.abs(i2),r2d*np.angle(i2),np.abs(3*i0),r2d*np.angle(i0)))
                    print("IA  = {0:>7.0f} < {1:>6.1f} | IB  = {2:>7.0f} < {3:>6.1f} | IC  = {4:>7.0f} < {5:>6.1f}".format(np.abs(iA),r2d*np.angle(iA),np.abs(iB),r2d*np.angle(iB),np.abs(iC),r2d*np.angle(iC)))
            else:
                res = mselectz[parname]([u1,u2,u0],[i1,i2,i0])
                if isinstance(res, (tuple, list)):
                    return mform3[subpar](res,parname)
                else:
                    return mform1[subpar](res,parname)

    def __repr__(self):
        '''
        Еще один способ вывода сводной таблицы результатов расчетов для несимметрии n
        В командной строке интерпретара набрать название переменной объекта несимметрии n и нажать Enter
        n Enter
        '''
        if isinstance(self.qp, Q):
            u1,u2,u0 = self.qp.getres()
            i1,i2,i0 = self.getres()
            iA,iB,iC,iAB,iBC,iCA = msymm2faze(i1,i2,i0)
            uA,uB,uC,uAB,uBC,uCA = msymm2faze(u1,u2,u0)
            strres = ('КЗ №{} - {} - {}\n'.format(self.id, self.name, self.SC))
            strres += ('В Узле № {} - {}\n'.format(self.qp.id, self.qp.name))
            strres += ("U1  = {0:>7.0f} < {1:>6.1f} | U2  = {2:>7.0f} < {3:>6.1f} | 3U0 = {4:>7.0f} < {5:>6.1f}\n".format(np.abs(u1),r2d*np.angle(u1),np.abs(u2),r2d*np.angle(u2),np.abs(3*u0),r2d*np.angle(u0)))
            strres += ("UA  = {0:>7.0f} < {1:>6.1f} | UB  = {2:>7.0f} < {3:>6.1f} | UC  = {4:>7.0f} < {5:>6.1f}\n".format(np.abs(uA),r2d*np.angle(uA),np.abs(uB),r2d*np.angle(uB),np.abs(uC),r2d*np.angle(uC)))
            strres += ("UAB = {0:>7.0f} < {1:>6.1f} | UBC = {2:>7.0f} < {3:>6.1f} | UCA = {4:>7.0f} < {5:>6.1f}\n".format(np.abs(uAB),r2d*np.angle(uAB),np.abs(uBC),r2d*np.angle(uBC),np.abs(uCA),r2d*np.angle(uCA)))
            strres += ('Суммарный ток КЗ в Узле № {} - {}\n'.format(self.qp.id, self.qp.name))
            strres += ("I1  = {0:>7.0f} < {1:>6.1f} | I2  = {2:>7.0f} < {3:>6.1f} | 3I0 = {4:>7.0f} < {5:>6.1f}\n".format(np.abs(i1),r2d*np.angle(i1),np.abs(i2),r2d*np.angle(i2),np.abs(3*i0),r2d*np.angle(i0)))
            strres += ("IA  = {0:>7.0f} < {1:>6.1f} | IB  = {2:>7.0f} < {3:>6.1f} | IC  = {4:>7.0f} < {5:>6.1f}\n".format(np.abs(iA),r2d*np.angle(iA),np.abs(iB),r2d*np.angle(iB),np.abs(iC),r2d*np.angle(iC)))
            strres += ("IAB = {0:>7.0f} < {1:>6.1f} | IBC = {2:>7.0f} < {3:>6.1f} | ICA = {4:>7.0f} < {5:>6.1f}\n".format(np.abs(iAB),r2d*np.angle(iAB),np.abs(iBC),r2d*np.angle(iBC),np.abs(iCA),r2d*np.angle(iCA)))
            strres += ('Подтекание токов по ветвям\n')

            for kp in self.qp.plist:
                i1,i2,i0 = kp.getres()
                if self.qp is kp.q1:
                    u1,u2,u0,i1,i2,i0 = kp.getresq1(i1,i2,i0)
                elif self.qp is kp.q2:
                    u1,u2,u0,i1,i2,i0 = kp.getresq2(i1,i2,i0)
                i1 = -i1
                i2 = -i2
                i0 = -i0
                iA = i1 + i2 + i0
                iB = a2*i1 + a*i2 + i0
                iC = a*i1 + a2*i2 + i0
                strres += ('Ветвь № {} - {}\n'.format(kp.id, kp.name))
                strres += ("I1  = {0:>7.0f} < {1:>6.1f} | I2  = {2:>7.0f} < {3:>6.1f} | 3I0 = {4:>7.0f} < {5:>6.1f}\n".format(np.abs(i1),r2d*np.angle(i1),np.abs(i2),r2d*np.angle(i2),np.abs(3*i0),r2d*np.angle(i0)))
                strres += ("IA  = {0:>7.0f} < {1:>6.1f} | IB  = {2:>7.0f} < {3:>6.1f} | IC  = {4:>7.0f} < {5:>6.1f}\n".format(np.abs(iA),r2d*np.angle(iA),np.abs(iB),r2d*np.angle(iB),np.abs(iC),r2d*np.angle(iC)))
        elif isinstance(self.qp, P):
            strres = self.qp.__repr__()
        return (strres)

    def __getattr__(self, attrname):
        '''
        Еще один способ получения конкректного параметра результата в виде
        компексного числа для его последующего использования в расчетах
        n.ParName
        где ParName может принимать значения:
        U1,U2,U0,UA,UB,UC,UABC,UAB,UBC,UCA,UAB_BC_CA
        I1,I2,I0,IA,IB,IC,IABC,IAB,IBC,ICA,IAB_BC_CA
        Z1,Z2,Z0,Z120,ZA,ZB,ZC,ZABC,ZAB,ZBC,ZCA,ZAB_BC_CA,
        S1,S2,S0,S120,SA,SB,SC,SABC,SAB,SBC,SCA,SAB_BC_CA,S
        '''
        if isinstance(self.qp, Q):
            u1,u2,u0 = self.qp.getres()
            i1,i2,i0 = self.getres()
        elif isinstance(self.qp, P):
            u1,u2,u0 = self.getres()
            i1,i2,i0 = self.qp.getres()
        res = mselectz[attrname]([u1,u2,u0],[i1,i2,i0])
        return res

class Model:
    '''
    Класс представляющий расчетную модель электрической сети,
    необходим для формирования и хранения расчетной модели, выполнения расчетов

    Конструктор расчетной модели сети
    Model()
    Model(desc='Примечание')


    Пользовательские функции для модели mdl
    Обнуление количества и очистка списков (таблиц) узлов, ветвей,
    взаимоиндукций, несимметрий...
    mdl.Clear()

    '''
    def __init__(self,desc=''):
        ''' Конструктор расчетной модели'''
        self.desc=desc
        self.nq=0
        self.np=0
        self.nm=0
        self.nn=0
        self.bq=[]
        self.bp=[]
        self.bm=[]
        self.bn=[]
        self.X = None

    def AddNQ(self,NQ,Nname):
        '''
        Множественное создание узлов
        NQ - количество создаваемых узлов
        Nname - общее наименование узлов
        '''
        listq = []
        for ij in range(NQ):
            listq.append(Q(self,'{} - №{}'.format(Nname,ij+1)))
        return listq

    def AddNP(self,Nname,listq1,listq2,Z12,Z0,B12=None,B0=None):
        '''
        Множественное создание ветвей и взаимоиндуктивностей
        NQ - количество создаваемых узлов
        Nname - общее наименование сечения ветвей
        listq1 - список объектов узлов к которым буду подключаться ветви
        listq2 - другой список объектов узлов к которым буду подключаться ветви
        Z12 - вектор np.ndarray значений сопротивлений ветвей прямой/обратной последовательности
        Z0 - квадратная матрица np.ndarray значений сопротивлений ветвей и взаимоиндукций нулевой последовательности
        B12 - вектор np.ndarray значений поперечной емкостной проводимости прямой/обратной последовательности
        B0 - квадратная матрица np.ndarray значений поперечной емкостной проводимости нулевой последовательности
        AddNP(Nname,listq1,listq2,Z12,Z0) - при отсутствии поперечной емкостной проводимости
        AddNP(Nname,listq1,listq2,Z12,Z0,B12,B0) - при наличии поперечной емкостной проводимости
        '''
        listp = []
        listm = []
        nq1 = len(listq1)
        nq2 = len(listq2)
        if nq1 != nq2:
            raise ValueError('Ошибка при добавлении сечения ветвей -', Nname, '\n',
                            'Количество узлов с обоих сторон должно совпадать!')
        if not isinstance(Z12, np.ndarray):
            raise TypeError('Ошибка при добавлении сечения ветвей -', Nname, '\n',
                            'Аргумент Z12 должен иметь тип np.ndarray!')
        if not isinstance(Z0, np.ndarray):
            raise TypeError('Ошибка при добавлении сечения ветвей -', Nname, '\n',
                            'Аргумент Z0 должен иметь тип np.ndarray!')
        if nq1 != Z12.shape[0]:
            raise ValueError('Ошибка при добавлении сечения ветвей -', Nname, '\n',
                            'Количество сопротивлений Z12 должно соответствовать количеству узлов!')
        if nq1 != Z0.shape[0] or nq1 != Z0.shape[1]:
            raise ValueError('Ошибка при добавлении сечения ветвей -', Nname, '\n',
                            'Количество сопротивлений Z0 должно соответствовать количеству узлов!')
        if isinstance(B12, np.ndarray) and isinstance(B0, np.ndarray):
            if not isinstance(B12, np.ndarray):
                raise TypeError('Ошибка при добавлении сечения ветвей -', Nname, '\n',
                                'Аргумент B12 должен иметь тип np.ndarray!')
            if not isinstance(B0, np.ndarray):
                raise TypeError('Ошибка при добавлении сечения ветвей -', Nname, '\n',
                                'Аргумент B0 должен иметь тип np.ndarray!')
            if nq1 != B12.shape[0]:
                raise ValueError('Ошибка при добавлении сечения ветвей -', Nname, '\n',
                                'Количество сопротивлений B12 должно соответствовать количеству узлов!')
            if nq1 != B0.shape[0] or nq1 != B0.shape[1]:
                raise ValueError('Ошибка при добавлении сечения ветвей -', Nname, '\n',
                                'Количество сопротивлений B0 должно соответствовать количеству узлов!')
            for ij in range(nq1):
                listp.append(P(self,'{} - №{}'.format(Nname,ij+1),listq1[ij],listq2[ij]),(Z12[ij],Z12[ij],Z0[ij,ij]),B=(B12[ij],B12[ij],B0[ij,ij]))
                for ij2 in range(ij):
                    listm.append(M(self,'{} - №{}<=>№{}'.format(Nname,ij+1,ij2+1),listp[ij],listp[ij2],Z0[ij,ij2],Z0[ij2,ij]))
        else:
            for ij in range(nq1):
                listp.append(P(self,'{} - №{}'.format(Nname,ij+1),listq1[ij],listq2[ij]),(Z12[ij],Z12[ij],Z0[ij,ij]))
                for ij2 in range(ij):
                    listm.append(M(self,'{} - №{}<=>№{}'.format(Nname,ij+1,ij2+1),listp[ij],listp[ij2],Z0[ij,ij2],Z0[ij2,ij]))
        return listp + listm

    def ImportFromPVL(self,PVL_Sech):
        '''Импорт сечений ветвей из PVL'''
        listp = []
        listm = []
        PVL_Sech.calc()
        z1 = PVL_Sech.Len * PVL_Sech.Z1
        z0 = PVL_Sech.Len * PVL_Sech.Z0
        b1 = PVL_Sech.Len * PVL_Sech.B1
        b0 = PVL_Sech.Len * PVL_Sech.B0
        for ij,pk in enumerate(PVL_Sech.bp):
            p1 = P(self, pk.name, pk.q1, pk.q2,
                   (z1[ij,0],z1[ij,0],z0[ij,ij]),
                   B=(b1[ij,0],b1[ij,0],b0[ij,ij]) )
            listp.append(p1)
            for ij2,pk2 in enumerate(PVL_Sech.bp[0:ij]):
                mname = '{} - №{}<=>№{}'.format(PVL_Sech.name,pk.name,pk2.name)
                p2 = listp[ij2]
                m = M(self,mname,p1,p2,z0[ij,ij2],z0[ij2,ij])
                listm.append(m)
        return listp + listm

    def Clear(self):
        '''Полная очистка расчетной модели
        Обнуление количества и очистка списков (таблиц) узлов, ветвей,
        взаимоиндукций, несимметрий...
        mdl.Clear()
        '''
        self.nq=0
        self.np=0
        self.nm=0
        self.nn=0
        for kq in self.bq:
            kq.model=None
        for kp in self.bp:
            kp.model=None
        for km in self.bm:
            km.model=None
        for kn in self.bn:
            kn.model=None
        self.bq=[]
        self.bp=[]
        self.bm=[]
        self.bn=[]

    def ClearN(self):
        '''Очистка всех несимметрий (КЗ и обрывов) в расчетной модели
        за исключением типа 'N0' - заземлений и обрывов по нулевой последовательности
        mdl.ClearN()
        '''
        self.nn=0
        oldbn=self.bn
        self.bn=[]
        for kn in oldbn:
            if kn.SC == 'N0':
                self.nn+=1
                self.bn.append(kn)
                kn.id = self.nn
            else:
                kn.model=None
                kn.qp.kn=None

    def List(self):
        '''Вывод на экран составляющих расчетную модель узлов, ветвей,
        взаимоиндукций, несимметрий и их параметров...
        По сути является поочередным применением метода par() ко всем элементам
        расчетной модели
        mdl.List()
        '''
        print('Количество узлов = {}; ветвей = {}; взаимоиндуктивностей = {}; несимметрий = {}'.format(self.nq,self.np,self.nm,self.nn))
        for kq in self.bq:
            kq.par()
        for kp in self.bp:
            kp.par()
        for km in self.bm:
            km.par()
        for kn in self.bn:
            kn.par()


    def Calc(self):
        '''Главный метод модуля МРТКЗ mdl.Calc()
        Осуществляет формирование разреженной системы линейных алгебраических уравнений (СЛАУ)
        и последующее ее решение с помощью алгоритма библиотеки scipy - spsolve(LHS,RHS)
        LHS * X = RHS
        где LHS - разреженная квадратная матрица
            RHS - вектор столбец
            X - искомый результат расчета
        Размерность (количество уравнений) равняется 3*(np+nq+nn), где:
            np - количество ветвей в расчетной модели;
            nq - количество узлов в расчетной модели;
            nn - количество несимметрий в расчетной модели.
        Вышеуказанное уравнение представляет собой систему матричных уравнений без учета несимметрий:
            Z*Ip + At*Uq = E
            A*Ip + (-B/2)*Uq = 0
            где:
                Z - квадратная матрица сопротивлений ветвей и взаимных индуктивностей
                прямой, обратной и нулевой последовательностей, размерность - (3*np,3*np)
                A - матрица соединений размерность - (3*nq,3*np)
                At - транспонированная матрица соединений размерность - (3*np,3*nq)
                (B/2) - квадратная диагональная матрица сумм поперечных проводимостей B/2,
                подключенных к узлу прямой, обратной и нулевой последовательностей, размерность - (3*nq,3*nq)
                E - вектор столбец Э.Д.С. ветвей прямой, обратной и нулевой последовательностей, размерность - (3*np,1)
                Ip - искомый вектор столбец значений токов ветвей
                прямой, обратной и нулевой последовательностей, размерность - (3*np,1)
                Uq - искомый вектор столбец значений напряжений узлов
                прямой, обратной и нулевой последовательностей, размерность - (3*nq,1)
            На каждую несимметрию дополнительно пишется по три уравнения -
            краевых условия (для понимания указаны в фазных величинах):
                Короткие замыкания
                А0 => Uka=0;Ikb=0;Ikc=0
                B0 => Ukb=0;Ikc=0;Ika=0
                C0 => Ukc=0;Ika=0;Ikb=0

                А0r => Uka-r*Ika=0;Ikb=0;Ikc=0
                B0r => Ukb-r*Ikb=0;Ikc=0;Ika=0
                C0r => Ukc-r*Ikc=0;Ika=0;Ikb=0

                АB => Uka-Ukb=0;Ika+Ikb=0;Ikc=0
                BC => Ukb-Ukc=0;Ikb+Ikc=0;Ika=0
                CА => Ukc-Uka=0;Ikc+Ika=0;Ikb=0

                АBr => Uka-Ukb-r*Ika=0;Ika+Ikb=0;Ikc=0
                BCr => Ukb-Ukc-r*Ikb=0;Ikb+Ikc=0;Ika=0
                CАr => Ukc-Uka-r*Ikc=0;Ikc+Ika=0;Ikb=0

                АB0 => Uka=0;Ukb=0;Ikc=0
                BC0 => Ukb=0;Ukc=0;Ika=0
                CА0 => Ukc=0;Uka=0;Ikb=0

                АBC => Uk1=0;Uk2=0;Ik0=0
                АBC0 => Uk1=0;Uk2=0;Uk0=0
                Заземление нейтрали => Ik1=0;Ik2=0;Uk0=0

                Обрывы
                А0 => Ia=0;dUb=0;dUc=0
                B0 => Ib=0;dUc=0;dUa=0
                C0 => Ic=0;dUa=0;dUb=0

                АB => Ia=0;Ib=0;dUc=0
                BC => Ib=0;Ic=0;dUa=0
                CА => Ic=0;Ia=0;dUb=0

                АBC => I1=0;I2=0;I0=0
                Обрыв ветви по нулевой последовательности  => dU1=0;dU2=0;I0=0
            а также в новых столбцах по каждой из последовательностей прописывается:
                - Для КЗ в уравнение по 1-ому закону Кирхгофа
                A*Ip + (-B/2)*Uq - Ik = 0, где Ik - ток поперечной несимметрии
                - Для обрывов в уравнение по 2-ому закону Кирхгофа
                Z*Ip + At*Uq + dU = E, где dU - напряжение продольной несимметрии

        Разреженная матрица LHS формируется в два этапа
        Этап 1. формируется координатная версия резреженной матрицы в списках
        cdata, ri и ci в которых хранятся значения ненулевых элеметнов матрицы
        и их номера строк и столбцов
        Этап 2. формируется CSC (Разреженный столбцовый формат) матрица LHS  с помощью метода scipy
        Решение разреженной СЛАУ осуществляется с помощью метода spsolve(LHS,RHS) библиотеки scipy
        '''
        n=3*(self.nq+self.np+self.nn)# Размерность СЛАУ
        RHS=np.zeros(n, dtype=complex)# Вектор для суммирования B/2 подключенных ветвей к узлу
        qB=np.zeros(3*self.nq, dtype=complex)# Временный вектор для расчета сумм B/2 ветвей подключенных к узлу

        cdata=[]# Список для хранения ненулевых элементов СЛАУ
        ri=[]# Список для хранения номеров строк ненулевых элементов СЛАУ
        ci=[]# Список для хранения номеров столбцов ненулевых элементов СЛАУ

        for kp in self.bp:
            pId=3*(kp.id-1)#Здесь и далее номер строки, столбца относящегося к прямой последовательности ветви
            lpId=[pId,pId+1,pId+2]
            #Запись сопротивлений ветви в разреженную матрицу
            ri+=lpId#[pId,pId+1,pId+2]
            ci+=lpId#[pId,pId+1,pId+2]
            cdata+=list(kp.Z)
            #Запись Э.Д.С. ветви в RHS
            RHS[pId]=kp.E[0]
            RHS[pId+1]=kp.E[1]
            RHS[pId+2]=kp.E[2]
            #Расчет комплексных коэф-ов трансформации прямой, обратной и нулевой последовательностей
            Kt1=kp.T[0]*np.exp(Kf*kp.T[1])
            if (kp.T[1]%2==0):
                Kt2=Kt1
            else:
                Kt2=np.conj(Kt1)
            Kt0=Kt1

            if isinstance(kp.q1, Q):
                qId=3*(self.np+kp.q1.id-1)#Здесь и далее номер строки, столбца относящегося к прямой последовательности узла
                lqId=[qId,qId+1,qId+2]
                qbId=3*(kp.q1.id-1)
                #Cуммирование B/2 подключенных ветвей к узлу
                qB[qbId]+=kp.B[0]/2
                qB[qbId+1]+=kp.B[1]/2
                qB[qbId+2]+=kp.B[2]/2
                #Запись матриц соединений A и At в разреженную матрицу (для q1 -> -1)
                ri+=lpId+lqId#[pId,pId+1,pId+2,qId,qId+1,qId+2]
                ci+=lqId+lpId#[qId,qId+1,qId+2,pId,pId+1,pId+2]
                cdata+=v_A+v_A#[-1.0,-1.0,-1.0,-1.0,-1.0,-1.0]

            if isinstance(kp.q2, Q):
                qId=3*(self.np+kp.q2.id-1)
                lqId=[qId,qId+1,qId+2]
                qbId=3*(kp.q2.id-1)
                #Cуммирование B/2 подключенных ветвей к узлу
                qB[qbId]+=kp.B[0]/2
                qB[qbId+1]+=kp.B[1]/2
                qB[qbId+2]+=kp.B[2]/2
                #Запись матриц соединений A и At в разреженную матрицу (для q2 -> 1 или Кт для трансформаторов)
                ri+=lpId+lqId#[pId,pId+1,pId+2,qId,qId+1,qId+2]
                ci+=lqId+lpId#[qId,qId+1,qId+2,pId,pId+1,pId+2]
                cdata+=[Kt2,Kt1,Kt0,Kt1,Kt2,Kt0]

        for km in self.bm:
            pId1=3*(km.p1.id-1)+2
            pId2=3*(km.p2.id-1)+2
            #Запись сопротивлений взаимоиндукции в разреженную матрицу
            ri+=[pId1,pId2]
            ci+=[pId2,pId1]
            cdata+=[km.M12,km.M21]

        for kq in self.bq:
            qId = 3*(self.np+kq.id-1)
            lqId=[qId,qId+1,qId+2]
            qbId = 3*(kq.id-1)
            #Запись сумм B/2 подключенных к узлу ветвей в разреженную матрицу
            ri+=lqId#[qId,qId+1,qId+2]
            ci+=lqId#[qId,qId+1,qId+2]
            cdata+=[-qB[qbId],-qB[qbId+1],-qB[qbId+2]]

        for kn in self.bn:
            nId = 3*(self.nq+self.np+kn.id-1)#Здесь и далее номер строки, столбца относящегося к несимметрии
            if isinstance(kn.qp, Q): # Короткие замыкания
                qId=3*(self.np+kn.qp.id-1);
                #Запись в разреженную матрицу в уравнения по 1-ому закону Кирхгофа наличие КЗ в узле
                ri+=[qId,qId+1,qId+2]
                ci+=[nId,nId+1,nId+2]
                cdata+=v_A#[-1.0,-1.0,-1.0]
                if kn.SC in ('A0','B0','C0'):
                    #Запись в разреженную матрицу граничных условий для КЗ
                    ri+=[nId,nId,nId,nId+1,nId+1,nId+1,nId+2,nId+2,nId+2]
                    ci+=[qId,qId+1,qId+2,nId,nId+1,nId+2,nId,nId+1,nId+2]
                    if kn.SC=='A0':# Uka=0;Ikb=0;Ikc=0
                        cdata+=vA+vB+vC#[1.0,1.0,1.0,a2,a,1.0,a,a2,1.0]
                    elif kn.SC=='B0':# Ukb=0;Ikc=0;Ika=0
                        cdata+=vB+vC+vA#[a2,a,1.0,1.0,1.0,1.0,a,a2,1.0]
                    else : # 'C0' # Ukc=0;Ika=0;Ikb=0
                        cdata+=vC+vA+vB#[a,a2,1.0,1.0,1.0,1.0,a2,a,1.0]
                elif kn.SC in ('A0r','B0r','C0r'):
                    ri+=[nId,nId,nId,nId+1,nId+1,nId+1,nId+2,nId+2,nId+2,nId,nId,nId]
                    ci+=[qId,qId+1,qId+2,nId,nId+1,nId+2,nId,nId+1,nId+2,nId,nId+1,nId+2]
                    if kn.SC=='A0r':# Uka-r*Ika=0;Ikb=0;Ikc=0
                        cdata+=vA+vB+vC+[-kn.r,-kn.r,-kn.r]#[1.0,1.0,1.0,a2,a,1.0,a,a2,1.0]
                    elif kn.SC=='B0r':# Ukb-r*Ikb=0;Ikc=0;Ika=0
                        cdata+=vB+vC+vA+[-kn.r*a2,-kn.r*a,-kn.r]#[a2,a,1.0,1.0,1.0,1.0,a,a2,1.0]
                    else : # 'C0r'# Ukc-r*Ikc=0;Ika=0;Ikb=0
                        cdata+=vC+vA+vB+[-kn.r*a,-kn.r*a2,-kn.r]#[a,a2,1.0,1.0,1.0,1.0,a2,a,1.0]
                elif kn.SC in ('AB','BC','CA'):
                    ri+=[nId,nId,nId+1,nId+1,nId+2]
                    ci+=[qId,qId+1,nId,nId+1,nId+2]
                    if kn.SC=='AB':# Uka-Ukb=0;Ika+Ikb=0;Ikc=0
                        cdata+=[1.0-a2,1.0-a,1.0+a2,1.0+a,1.0]
                    elif kn.SC=='BC':# Ukb-Ukc=0;Ikb+Ikc=0;Ika=0
                        cdata+=[a2-a,a-a2,a2+a,a+a2,1.0]
                    else : # 'CA'# Ukc-Uka=0;Ikc+Ika=0;Ikb=0
                        cdata+=[a-1.0,a2-1.0,a+1.0,a2+1.0,1.0]
                elif kn.SC in ('ABr','BCr','CAr'):
                    ri+=[nId,nId,nId+1,nId+1,nId+2,nId,nId]
                    ci+=[qId,qId+1,nId,nId+1,nId+2,nId,nId+1]
                    if kn.SC=='ABr':# Uka-Ukb-r*Ika=0;Ika+Ikb=0;Ikc=0
                        cdata+=[1.0-a2,1.0-a,1.0+a2,1.0+a,1.0,-kn.r,-kn.r]
                    elif kn.SC=='BCr':# Ukb-Ukc-r*Ikb=0;Ikb+Ikc=0;Ika=0
                        cdata+=[a2-a,a-a2,a2+a,a+a2,1.0,-kn.r*a2,-kn.r*a]
                    else : # 'CAr'# Ukc-Uka-r*Ikc=0;Ikc+Ika=0;Ikb=0
                        cdata+=[a-1.0,a2-1.0,a+1.0,a2+1.0,1.0,-kn.r*a,-kn.r*a2]
                elif kn.SC in ('AB0','BC0','CA0'):
                    ri+=[nId,nId,nId,nId+1,nId+1,nId+1,nId+2,nId+2,nId+2]
                    ci+=[qId,qId+1,qId+2,qId,qId+1,qId+2,nId,nId+1,nId+2]
                    if kn.SC=='AB0':# Uka=0;Ukb=0;Ikc=0
                        cdata+=vA+vB+vC#[1.0,1.0,1.0,a2,a,1.0,a,a2,1.0]
                    elif kn.SC=='BC0':# Ukb=0;Ukc=0;Ika=0
                        cdata+=vB+vC+vA#[a2,a,1.0,a,a2,1.0,1.0,1.0,1.0]
                    else : # 'CA0'# Ukc=0;Uka=0;Ikb=0
                        cdata+=vC+vA+vB#[a,a2,1.0,1.0,1.0,1.0,a2,a,1.0]
                elif kn.SC=='ABC':# Uk1=0;Uk2=0;Ik0=0
                    ri+=[nId,nId+1,nId+2]
                    ci+=[qId,qId+1,nId+2]
                    cdata+=vA#[1.0,1.0,1.0]
                elif kn.SC=='ABC0' : # Uk1=0;Uk2=0;Uk0=0
                    ri+=[nId,nId+1,nId+2]
                    ci+=[qId,qId+1,qId+2]
                    cdata+=vA#[1.0,1.0,1.0]
                elif kn.SC=='N0' : #Заземление нейтрали Ik1=0;Ik2=0;Uk0=0
                    ri+=[nId,nId+1,nId+2]
                    ci+=[nId,nId+1,qId+2]
                    cdata+=vA#[1.0,1.0,1.0]
                else :
                    raise TypeError('Неизвестный вид КЗ!')

            elif  isinstance(kn.qp, P): #Обрывы
                pId=3*(kn.qp-1)
                #Запись в разреженную матрицу в уравнения по 2-ому закону Кирхгофа наличие обрыва на ветви
                ri+=[pId,pId+1,pId+2]
                ci+=[nId,nId+1,nId+2]
                cdata+=[1.0,1.0,1.0]
                if kn.SC in ('A0','B0','C0'):
                    ri+=[nId,nId,nId,nId+1,nId+1,nId+1,nId+2,nId+2,nId+2]
                    ci+=[pId,pId+1,pId+2,nId,nId+1,nId+2,nId,nId+1,nId+2]
                    if kn.SC=='A0':# Ia=0;dUb=0;dUc=0
                        cdata+=vA+vB+vC#[1.0,1.0,1.0,a2,a,1.0,a,a2,1.0]
                    elif kn.SC=='B0':# Ib=0;dUc=0;dUa=0
                        cdata+=vB+vC+vA#[a2,a,1.0,a,a2,1.0,1.0,1.0,1.0]
                    else : # 'C0'# Ic=0;dUa=0;dUb=0
                        cdata+=vC+vA+vB#[a,a2,1.0,1.0,1.0,1.0,a2,a,1.0]
                elif kn.SC in ('AB','BC','CA'):
                    ri+=[nId,nId,nId,nId+1,nId+1,nId+1,nId+2,nId+2,nId+2]
                    ci+=[pId,pId+1,pId+2,pId,pId+1,pId+2,nId,nId+1,nId+2]
                    if kn.SC=='AB':# Ia=0;Ib=0;dUc=0
                        cdata+=vA+vB+vC#[1.0,1.0,1.0,a2,a,1.0,a,a2,1.0]
                    elif kn.SC=='BC':# Ib=0;Ic=0;dUa=0
                        cdata+=vB+vC+vA#[a2,a,1.0,a,a2,1.0,1.0,1.0,1.0]
                    else : # 'CA'# Ic=0;Ia=0;dUb=0
                        cdata += vC + vA + vB#[a,a2,1.0,1.0,1.0,1.0,a2,a,1.0]
                elif kn.SC == 'ABC'  : # I1=0;I2=0;I0=0
                    ri += [nId, nId+1, nId+2]
                    ci += [pId, pId+1, pId+2]
                    cdata += vA#[1.0,1.0,1.0]
                elif kn.SC == 'N0': #Обрыв ветви по нулевой последовательности dU1=0;dU2=0;I0=0
                    ri += [nId, nId+1, nId+2]
                    ci += [nId, nId+1, pId+2]
                    cdata += vA#[1.0,1.0,1.0]
                else: raise TypeError('Неизвестный вид обрыва!')
            else: raise TypeError('Неизвестный вид несимметрии!')

        #Преобразование списков python3 в вектора numpy (numpy.ndarray)
        row = np.array(ri)
        col = np.array(ci)
        data = np.array(cdata)
        #Формирование разреженной матрицы
        LHS = csc_matrix((data, (row, col)), shape=(n, n))
        #решение разреженной СЛАУ с помощью функции из состава scipy
        self.X = spsolve(LHS,RHS)
        return self.X

mselectz=dict({'U120' : lambda uq,ip: uq,
              'U1' : lambda uq,ip: uq[0],
              'U2' : lambda uq,ip: uq[1],
              'U0' : lambda uq,ip: uq[2],
              '3U0' : lambda uq,ip: 3*uq[2],
              'UA' : lambda uq,ip: uq[0]+uq[1]+uq[2],
              'UB' : lambda uq,ip: a2*uq[0]+a*uq[1]+uq[2],
              'UC' : lambda uq,ip: a*uq[0]+a2*uq[1]+uq[2],
              'UAB' : lambda uq,ip: (1-a2)*uq[0]+(1-a)*uq[1],
              'UBC' : lambda uq,ip: (a2-a)*uq[0]+(a-a2)*uq[1],
              'UCA' : lambda uq,ip: (a-1)*uq[0]+(a2-1)*uq[1],
              'UABC' : lambda uq,ip: (uq[0]+uq[1]+uq[2],
                                         a2*uq[0]+a*uq[1]+uq[2],
                                         a*uq[0]+a2*uq[1]+uq[2]),
              'UAB_BC_CA' : lambda uq,ip: ((1-a2)*uq[0]+(1-a)*uq[1],
                                              (a2-a)*uq[0]+(a-a2)*uq[1],
                                              (a-1)*uq[0]+(a2-1)*uq[1]),
              'I120' : lambda uq,ip: ip,
              'I1' : lambda uq,ip: ip[0],
              'I2' : lambda uq,ip: ip[1],
              'I0' : lambda uq,ip: ip[2],
              '3I0' : lambda uq,ip: 3*ip[2],
              'IA' : lambda uq,ip: ip[0]+ip[1]+ip[2],
              'IB' : lambda uq,ip: a2*ip[0]+a*ip[1]+ip[2],
              'IC' : lambda uq,ip: a*ip[0]+a2*ip[1]+ip[2],
              'IAB' : lambda uq,ip: (1-a2)*ip[0]+(1-a)*ip[1],
              'IBC' : lambda uq,ip: (a2-a)*ip[0]+(a-a2)*ip[1],
              'ICA' : lambda uq,ip: (a-1)*ip[0]+(a2-1)*ip[1],
              'IABC' : lambda uq,ip: (ip[0]+ip[1]+ip[2],
                                         a2*ip[0]+a*ip[1]+ip[2],
                                         a*ip[0]+a2*ip[1]+ip[2]),
              'IAB_BC_CA' : lambda uq,ip: ((1-a2)*ip[0]+(1-a)*ip[1],
                                              (a2-a)*ip[0]+(a-a2)*ip[1],
                                              (a-1)*ip[0]+(a2-1)*ip[1]),
              'Z120' : lambda uq,ip: (uq[0]/ip[0],uq[1]/ip[1],uq[2]/ip[2]),
              'Z1' : lambda uq,ip: uq[0]/ip[0],
              'Z2' : lambda uq,ip: uq[1]/ip[1],
              'Z0' : lambda uq,ip: uq[2]/ip[2],
              'ZA' : lambda uq,ip: (uq[0]+uq[1]+uq[2])/(ip[0]+ip[1]+ip[2]),
              'ZB' : lambda uq,ip: (a2*uq[0]+a*uq[1]+uq[2])/(a2*ip[0]+a*ip[1]+ip[2]),
              'ZC' : lambda uq,ip: (a*uq[0]+a2*uq[1]+uq[2])/(a*ip[0]+a2*ip[1]+ip[2]),
              'ZAB' : lambda uq,ip: ((1-a2)*uq[0]+(1-a)*uq[1])/((1-a2)*ip[0]+(1-a)*ip[1]),
              'ZBC' : lambda uq,ip: ((a2-a)*uq[0]+(a-a2)*uq[1])/((a2-a)*ip[0]+(a-a2)*ip[1]),
              'ZCA' : lambda uq,ip: ((a-1)*uq[0]+(a2-1)*uq[1])/((a-1)*ip[0]+(a2-1)*ip[1]),
              'ZABC' : lambda uq,ip: ((uq[0]+uq[1]+uq[2])/(ip[0]+ip[1]+ip[2]),
                                         (a2*uq[0]+a*uq[1]+uq[2])/(a2*ip[0]+a*ip[1]+ip[2]),
                                         (a*uq[0]+a2*uq[1]+uq[2])/(a*ip[0]+a2*ip[1]+ip[2])),
              'ZAB_BC_CA' : lambda uq,ip: (((1-a2)*uq[0]+(1-a)*uq[1])/((1-a2)*ip[0]+(1-a)*ip[1]),
                                         ((a2-a)*uq[0]+(a-a2)*uq[1])/((a2-a)*ip[0]+(a-a2)*ip[1]),
                                         ((a-1)*uq[0]+(a2-1)*uq[1])/((a-1)*ip[0]+(a2-1)*ip[1])),
              'S120' : lambda uq,ip: (uq[0]*np.conj(ip[0]),uq[1]*np.conj(ip[1]),uq[2]*np.conj(ip[2])),
              'S1' : lambda uq,ip: uq[0]*np.conj(ip[0]),
              'S2' : lambda uq,ip: uq[1]*np.conj(ip[1]),
              'S0' : lambda uq,ip: uq[2]*np.conj(ip[2]),
              'SA' : lambda uq,ip: (uq[0]+uq[1]+uq[2])*np.conj(ip[0]+ip[1]+ip[2]),
              'SB' : lambda uq,ip: (a2*uq[0]+a*uq[1]+uq[2])*np.conj(a2*ip[0]+a*ip[1]+ip[2]),
              'SC' : lambda uq,ip: (a*uq[0]+a2*uq[1]+uq[2])*np.conj(a*ip[0]+a2*ip[1]+ip[2]),
              'SAB' : lambda uq,ip: ((1-a2)*uq[0]+(1-a)*uq[1])*np.conj((1-a2)*ip[0]+(1-a)*ip[1]),
              'SBC' : lambda uq,ip: ((a2-a)*uq[0]+(a-a2)*uq[1])*np.conj((a2-a)*ip[0]+(a-a2)*ip[1]),
              'SCA' : lambda uq,ip: ((a-1)*uq[0]+(a2-1)*uq[1])*np.conj((a-1)*ip[0]+(a2-1)*ip[1]),
              'SABC' : lambda uq,ip: ((uq[0]+uq[1]+uq[2])*np.conj(ip[0]+ip[1]+ip[2]),
                                         (a2*uq[0]+a*uq[1]+uq[2])*np.conj(a2*ip[0]+a*ip[1]+ip[2]),
                                         (a*uq[0]+a2*uq[1]+uq[2])*np.conj(a*ip[0]+a2*ip[1]+ip[2])),
              'S' : lambda uq,ip: ((uq[0]+uq[1]+uq[2])*np.conj(ip[0]+ip[1]+ip[2])+
                                         (a2*uq[0]+a*uq[1]+uq[2])*np.conj(a2*ip[0]+a*ip[1]+ip[2])+
                                         (a*uq[0]+a2*uq[1]+uq[2])*np.conj(a*ip[0]+a2*ip[1]+ip[2])),
              'SAB_BC_CA' : lambda uq,ip: (((1-a2)*uq[0]+(1-a)*uq[1])*np.conj((1-a2)*ip[0]+(1-a)*ip[1]),
                                         ((a2-a)*uq[0]+(a-a2)*uq[1])*np.conj((a2-a)*ip[0]+(a-a2)*ip[1]),
                                         ((a-1)*uq[0]+(a2-1)*uq[1])*np.conj((a-1)*ip[0]+(a2-1)*ip[1]))
              })

mform1=dict({'' : lambda res,parname: res,
              'R' : lambda res,parname: np.real(res),
              'X' : lambda res,parname: np.imag(res),
              'M' : lambda res,parname: np.abs(res),
              '<f' : lambda res,parname:  r2d*np.angle(res),
              'R+jX' : lambda res,parname: "{0:<4} = {1:>8.1f} + {2:>8.1f}j".format(parname, np.real(res),np.imag(res)),
              'M<f' : lambda res,parname: "{0:<4} = {1:>8.1f} < {2:>6.1f}".format(parname, np.abs(res),r2d*np.angle(res))
              })

mform3=dict({'' : lambda res,parname: res,
              'R' : lambda res,parname: [np.real(res[0]), np.real(res[1]), np.real(res[2])],
              'X' : lambda res,parname: [np.imag(res[0]), np.imag(res[1]), np.imag(res[2])],
              'M' : lambda res,parname: [np.abs(res[0]), np.abs(res[1]), np.abs(res[2])],
              '<f' : lambda res,parname:  [r2d*np.angle(res[0]), r2d*np.angle(res[1]), r2d*np.angle(res[2])],
              'R+jX' : lambda res,parname: "{0:<4} = [{1:>8.1f} + {2:>8.1f}j, {3:>8.1f} + {4:>8.1f}j, {5:>8.1f} + {6:>8.1f}j]".format(parname, np.real(res[0]), np.imag(res[0]), np.real(res[1]), np.imag(res[1]), np.real(res[2]), np.imag(res[2])),
              'M<f' : lambda res,parname: "{0:<4} = [{1:>8.1f} < {2:>6.1f}, {3:>8.1f} < {4:>6.1f}, {5:>8.1f} < {6:>6.1f}]".format(parname, np.abs(res[0]), r2d*np.angle(res[0]), np.abs(res[1]), r2d*np.angle(res[1]), np.abs(res[2]), r2d*np.angle(res[2]))
              })

def msymm2faze(p1,p2,p0):
    '''
    Служебная функция для расчета фазных и междуфазных (линейных) параметров
    напряжений и токов'''
    pA = p1 + p2 + p0
    pB = a2*p1 + a*p2 + p0
    pC = a*p1 + a2*p2 + p0
    pAB = pA - pB
    pBC = pB - pC
    pCA = pC - pA
    return [pA,pB,pC,pAB,pBC,pCA]
