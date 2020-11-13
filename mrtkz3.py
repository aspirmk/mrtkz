#Версия 3.04 26.10.2020
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

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
    def __init__(self,model,name,desc=''):
        ''' Конструктор узла'''
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
        self.plist.append(kp)

    def update(self):
        temp_plist = self.plist
        self.plist=[]
        for kp in temp_plist:
            if (kp.q1 is self) or (kp.q2 is self):
                self.plist.append(kp)

    def setn(self,kn):
        self.kn=kn

    def par(self):
        print('Узел №', self.id, ' - ', self.name)

    def getres(self):
        if self.model is None:
            raise ValueError('Ошибка при выводе результатов расчетов Узла №', self.id, ' - ', self.name, '\n',
                            'Узел не принадлежит какой либо модели!')
        if self.model.X is None:
            raise ValueError('Ошибка при выводе результатов расчетов Узла №', self.id, ' - ', self.name, '\n',
                            'Не произведен расчет электрических величин!')
        qId = 3*(self.model.np+self.id-1)
        return self.model.X[qId:qId+3]

    def res(self,parname='',subpar=''):
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
        u1,u2,u0 = self.getres()
        res = mselectz[attrname]([u1,u2,u0],[0j,0j,0j])
        return res

    def __repr__(self):
        u1,u2,u0 = self.getres()
        uA,uB,uC,uAB,uBC,uCA = msymm2faze(u1,u2,u0)
        strres  = "Узел № {} - {}\n".format(self.id, self.name)
        strres += "U1  = {0:>7.0f} < {1:>6.1f} | U2  = {2:>7.0f} < {3:>6.1f} | 3U0 = {4:>7.0f} < {5:>6.1f}\n".format(np.abs(u1),r2d*np.angle(u1),np.abs(u2),r2d*np.angle(u2),np.abs(3*u0),r2d*np.angle(u0))
        strres += "UA  = {0:>7.0f} < {1:>6.1f} | UB  = {2:>7.0f} < {3:>6.1f} | UC  = {4:>7.0f} < {5:>6.1f}\n".format(np.abs(uA),r2d*np.angle(uA),np.abs(uB),r2d*np.angle(uB),np.abs(uC),r2d*np.angle(uC))
        strres += "UAB = {0:>7.0f} < {1:>6.1f} | UBC = {2:>7.0f} < {3:>6.1f} | UCA = {4:>7.0f} < {5:>6.1f}".format(np.abs(uAB),r2d*np.angle(uAB),np.abs(uBC),r2d*np.angle(uBC),np.abs(uCA),r2d*np.angle(uCA))
        return (strres)



class P:
    def __init__(self,model,name,q1,q2,Z,E=(0, 0, 0),T=(1, 0),B=(0, 0, 0),desc=''):
        ''' Конструктор ветви'''
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
            raise Warning('Предупреждение при добавлении ветви -', name, '\n',
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
        ''' редактирование ветви'''
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
            raise Warning('Предупреждение при редактировании ветви №', self.id, ' - ', self.name, '\n',
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
        self.mlist.append(mid)

    def setn(self,kn):
        self.kn=kn

    def par(self):
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
        if self.model is None:
            raise ValueError('Ошибка при выводе результатов расчетов Ветви №', self.id, ' - ', self.name, '\n',
                            'Ветвь не принадлежит какой либо модели!')
        if self.model.X is None:
            raise ValueError('Ошибка при выводе результатов расчетов Ветви №', self.id, ' - ', self.name, '\n',
                            'Не произведен расчет электрических величин!')
        pId = 3*(self.id-1)
        return self.model.X[pId:pId+3]

    def getresq1(self,i1,i2,i0):
        if isinstance(self.q1, Q):
            u1,u2,u0 = self.q1.getres()
        else:
            u1,u2,u0 = [0j,0j,0j]
        i1 += u1 * self.B[0]
        i2 += u2 * self.B[1]
        i0 += u0 * self.B[2]
        return [u1,u2,u0,i1,i2,i0]

    def getresq2(self,i1,i2,i0):
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
        Kt0=Kt
        i1 = -Kt1*i1 + u1 * self.B[0]
        i2 = -Kt2*i2 + u2 * self.B[1]
        i0 = -Kt0*i0 + u0 * self.B[2]
        return [u1,u2,u0,i1,i2,i0]

    def res1(self,parname='',subpar=''):
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
    def __init__(self,model,name,p1,p2,M12,M21,desc=''):
        ''' Конструктор взаимоиндукции'''
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
        ''' Редактирование взаимоиндукции'''
        self.name=name
        self.M12=M12
        self.M21=M21

    def par(self):
        print('Взаимоиндукция № {} - {} : {}({}) <=> {}({})'.format(self.id,self.name,self.p1.id,self.p1.name,self.p2.id,self.p2.name))
        print('M12 = {}; M21 = {}'.format(self.M12,self.M21))

class N:
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

    def edit(self, name,SC,desc=''):
        ''' Редактирование повреждения (КЗ или обрыва)'''
        self.name=name
        self.desc=desc
        self.SC=SC

    def par(self):
        if isinstance(self.qp, Q):
            print('КЗ № {} - {} : {} (r={}) в узле № {}({})'.format(self.id,self.name,self.SC,self.r,self.qp.id,self.qp.name))
        elif isinstance(self.qp, P):
            print('Обрыв № {} - {} : {} на ветви № {}({})'.format(self.id,self.name,self.SC,self.qp.id,self.qp.name))

    def getres(self):
        if self.model is None:
            raise ValueError('Ошибка при выводе результатов расчетов несимметрии №', self.id, ' - ', self.name, '\n',
                            'Несимметрия не принадлежит какой либо модели!')
        if self.model.X is None:
            raise ValueError('Ошибка при выводе результатов расчетов несимметрии №', self.id, ' - ', self.name, '\n',
                            'Не произведен расчет электрических величин!')
        nId = 3*(self.model.np+self.model.nq+self.id-1)
        return self.model.X[nId:nId+3]

    def res(self,parname='',subpar=''):
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
        if isinstance(self.qp, Q):
            u1,u2,u0 = self.qp.getres()
            i1,i2,i0 = self.getres()
        elif isinstance(self.qp, P):
            u1,u2,u0 = self.getres()
            i1,i2,i0 = self.qp.getres()
        res = mselectz[attrname]([u1,u2,u0],[i1,i2,i0])
        return res

class Model:
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

    def ClearBase(self):
        '''Полная очистка расчетной модели'''
        self.nq=0
        self.np=0
        self.nm=0
        for kq in self.bq:
            kq.model=None
        for kp in self.bp:
            kp.model=None
        for km in self.bm:
            km.model=None
        self.bq=[]
        self.bp=[]
        self.bm=[]
        self.ClearN()

    def ClearN(self):
        '''Очистка всех несимметрий в расчетной модели'''
        for kn in self.bn:
            kn.model=None
            kn.qp.kn=None
        self.nn=0
        self.bn=[]

    def ListBase(self):
        '''Вывод расчетной модели'''
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
        '''Расчет электрических величин по расчетной модели'''
        n=3*(self.nq+self.np+self.nn)
        RHS=np.zeros(n, dtype=complex)
        qB=np.zeros(3*self.nq, dtype=complex)

        ri=[];ci=[];cdata=[]

        for kp in self.bp:
            pId=3*(kp.id-1)
            lpId=[pId,pId+1,pId+2]
            ri+=lpId#[pId,pId+1,pId+2]
            ci+=lpId#[pId,pId+1,pId+2]
            cdata+=list(kp.Z)

            RHS[pId]=kp.E[0]
            RHS[pId+1]=kp.E[1]
            RHS[pId+2]=kp.E[2]

            Kt1=kp.T[0]*np.exp(Kf*kp.T[1])
            if (kp.T[1]%2==0):
                Kt2=Kt1
            else :
                Kt2=np.conj(Kt1)
            Kt0=kp.T[0]

            if isinstance(kp.q1, Q):
                qId=3*(self.np+kp.q1.id-1)
                lqId=[qId,qId+1,qId+2]
                qbId=3*(kp.q1.id-1)
                qB[qbId]=kp.B[0]
                qB[qbId+1]=kp.B[1]
                qB[qbId+2]=kp.B[2]
                ri+=lpId+lqId#[pId,pId+1,pId+2,qId,qId+1,qId+2]
                ci+=lqId+lpId#[qId,qId+1,qId+2,pId,pId+1,pId+2]
                cdata+=v_A+v_A#[-1.0,-1.0,-1.0,-1.0,-1.0,-1.0]

            if isinstance(kp.q2, Q):
                qId=3*(self.np+kp.q2.id-1)
                lqId=[qId,qId+1,qId+2]
                qbId=3*(kp.q2.id-1)
                qB[qbId]=kp.B[0]
                qB[qbId+1]=kp.B[1]
                qB[qbId+2]=kp.B[2]
                ri+=lpId+lqId#[pId,pId+1,pId+2,qId,qId+1,qId+2]
                ci+=lqId+lpId#[qId,qId+1,qId+2,pId,pId+1,pId+2]
                cdata+=[Kt1,Kt2,Kt0,Kt1,Kt2,Kt0]

        for km in self.bm:
            pId1=3*(km.p1.id-1)+2
            pId2=3*(km.p2.id-1)+2
            ri+=[pId1,pId2]
            ci+=[pId2,pId1]
            cdata+=[km.M12,km.M21]

        for kq in self.bq:
            qId = 3*(self.np+kq.id-1)
            lqId=[qId,qId+1,qId+2]
            qbId = 3*(kq.id-1);
            ri+=lqId#[qId,qId+1,qId+2]
            ci+=lqId#[qId,qId+1,qId+2]
            cdata+=[qB[qbId],qB[qbId+1],qB[qbId+2]]

        for kn in self.bn:
            nId = 3*(self.nq+self.np+kn.id-1)
            if isinstance(kn.qp, Q): # Короткие замыкания
                qId=3*(self.np+kn.qp.id-1);
                ri+=[qId,qId+1,qId+2]
                ci+=[nId,nId+1,nId+2]
                cdata+=v_A#[-1.0,-1.0,-1.0]
                if kn.SC in ('A0','B0','C0'):
                    ri+=[nId,nId,nId,nId+1,nId+1,nId+1,nId+2,nId+2,nId+2]
                    ci+=[qId,qId+1,qId+2,nId,nId+1,nId+2,nId,nId+1,nId+2]
                    if kn.SC=='A0':
                        cdata+=vA+vB+vC#[1.0,1.0,1.0,a2,a,1.0,a,a2,1.0]
                    elif kn.SC=='B0':
                        cdata+=vB+vC+vA#[a2,a,1.0,1.0,1.0,1.0,a,a2,1.0]
                    else : # 'C0'
                        cdata+=vC+vA+vB#[a,a2,1.0,1.0,1.0,1.0,a2,a,1.0]
                elif kn.SC in ('A0r','B0r','C0r'):
                    ri+=[nId,nId,nId,nId+1,nId+1,nId+1,nId+2,nId+2,nId+2,nId,nId,nId]
                    ci+=[qId,qId+1,qId+2,nId,nId+1,nId+2,nId,nId+1,nId+2,nId,nId+1,nId+2]
                    if kn.SC=='A0r':
                        cdata+=vA+vB+vC+[-kn.r,-kn.r,-kn.r]#[1.0,1.0,1.0,a2,a,1.0,a,a2,1.0]
                    elif kn.SC=='B0r':
                        cdata+=vB+vC+vA+[-kn.r*a2,-kn.r*a,-kn.r]#[a2,a,1.0,1.0,1.0,1.0,a,a2,1.0]
                    else : # 'C0r'
                        cdata+=vC+vA+vB+[-kn.r*a,-kn.r*a2,-kn.r]#[a,a2,1.0,1.0,1.0,1.0,a2,a,1.0]
                elif kn.SC in ('AB','BC','CA'):
                    ri+=[nId,nId,nId+1,nId+1,nId+2]
                    ci+=[qId,qId+1,nId,nId+1,nId+2]
                    if kn.SC=='AB':
                        cdata+=[1.0-a2,1.0-a,1.0+a2,1.0+a,1.0]
                    elif kn.SC=='BC':
                        cdata+=[a2-a,a-a2,a2+a,a+a2,1.0]
                    else : # 'CA'
                        cdata+=[a-1.0,a2-1.0,a+1.0,a2+1.0,1.0]
                elif kn.SC in ('ABr','BCr','CAr'):
                    ri+=[nId,nId,nId+1,nId+1,nId+2,nId,nId]
                    ci+=[qId,qId+1,nId,nId+1,nId+2,nId,nId+1]
                    if kn.SC=='ABr':
                        cdata+=[1.0-a2,1.0-a,1.0+a2,1.0+a,1.0,-kn.r,-kn.r]
                    elif kn.SC=='BCr':
                        cdata+=[a2-a,a-a2,a2+a,a+a2,1.0,-kn.r*a2,-kn.r*a]
                    else : # 'CAr'
                        cdata+=[a-1.0,a2-1.0,a+1.0,a2+1.0,1.0,-kn.r*a,-kn.r*a2]
                elif kn.SC in ('AB0','BC0','CA0'):
                    ri+=[nId,nId,nId,nId+1,nId+1,nId+1,nId+2,nId+2,nId+2]
                    ci+=[qId,qId+1,qId+2,qId,qId+1,qId+2,nId,nId+1,nId+2]
                    if kn.SC=='AB0':
                        cdata+=vA+vB+vC#[1.0,1.0,1.0,a2,a,1.0,a,a2,1.0]
                    elif kn.SC=='BC0':
                        cdata+=vB+vC+vA#[a2,a,1.0,a,a2,1.0,1.0,1.0,1.0]
                    else : # 'CA0'
                        cdata+=vC+vA+vB#[a,a2,1.0,1.0,1.0,1.0,a2,a,1.0]
                else :
                    if kn.SC=='ABC':
                        ri+=[nId,nId+1,nId+2]
                        ci+=[qId,qId+1,nId+2]
                        cdata+=vA#[1.0,1.0,1.0]
                    else : #'ABC0'
                        ri+=[nId,nId+1,nId+2]
                        ci+=[qId,qId+1,qId+2]
                        cdata+=vA#[1.0,1.0,1.0]
            elif  isinstance(kn.qp, P): #Обрывы
                pId=3*(kn.qp-1)
                ri+=[pId,pId+1,pId+2]
                ci+=[nId,nId+1,nId+2]
                cdata+=[1.0,1.0,1.0]
                if kn.SC in ('A0','B0','C0'):
                    ri+=[nId,nId,nId,nId+1,nId+1,nId+1,nId+2,nId+2,nId+2]
                    ci+=[pId,pId+1,pId+2,nId,nId+1,nId+2,nId,nId+1,nId+2]
                    if kn.SC=='A0':
                        cdata+=vA+vB+vC#[1.0,1.0,1.0,a2,a,1.0,a,a2,1.0]
                    elif kn.SC=='B0':
                        cdata+=vB+vC+vA#[a2,a,1.0,a,a2,1.0,1.0,1.0,1.0]
                    else : # 'C0'
                        cdata+=vC+vA+vB#[a,a2,1.0,1.0,1.0,1.0,a2,a,1.0]
                elif kn.SC in ('AB','BC','CA'):
                    ri+=[nId,nId,nId,nId+1,nId+1,nId+1,nId+2,nId+2,nId+2]
                    ci+=[pId,pId+1,pId+2,pId,pId+1,pId+2,nId,nId+1,nId+2]
                    if kn.SC=='AB':
                        cdata+=vA+vB+vC#[1.0,1.0,1.0,a2,a,1.0,a,a2,1.0]
                    elif kn.SC=='BC':
                        cdata+=vB+vC+vA#[a2,a,1.0,a,a2,1.0,1.0,1.0,1.0]
                    else : # 'CA'
                        cdata += vC + vA + vB#[a,a2,1.0,1.0,1.0,1.0,a2,a,1.0]
                else : # 'ABC'
                    ri += [nId, nId+1, nId+2]
                    ci += [pId, pId+1, pId+2]
                    cdata += vA#[1.0,1.0,1.0]
            else: raise

        #Формирование разряженной матрицы
        row = np.array(ri)
        col = np.array(ci)
        data = np.array(cdata)
        LHS = csc_matrix((data, (row, col)), shape=(n, n))
        #решение разряженной сисетмы линейных алгебраических уравнений
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
    pA = p1 + p2 + p0
    pB = a2*p1 + a*p2 + p0
    pC = a*p1 + a2*p2 + p0
    pAB = pA - pB
    pBC = pB - pC
    pCA = pC - pA
    return [pA,pB,pC,pAB,pBC,pCA]
