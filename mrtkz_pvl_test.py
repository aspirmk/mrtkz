#Импортирование модуля расчета ТКЗ (mrtkz3.py) и
#модуля расчета параметров воздушных линий  PVL,
#которые должны находиться в той же папке, где и настоящий файл
import mrtkz3 as mrtkz
import PVL5 as PVL
#Создание расчетной модели
mdl=mrtkz.Model()

#Создание узлов
q1 = mrtkz.Q(mdl,'Sys1')
q2 = mrtkz.Q(mdl,'Sys2')
q3 = mrtkz.Q(mdl,'PS1 ВН')
q4 = mrtkz.Q(mdl,'PS2 ВН')
q5 = mrtkz.Q(mdl,'PS1 НН')
q6 = mrtkz.Q(mdl,'PS2 НН')

#Создание ветвей энергосистем
Sys1 = mrtkz.P(mdl,'Sys1',0,q1,(2j,2j,3j),E=(65000,0,0))
Sys2 = mrtkz.P(mdl,'Sys2',0,q2,(2j,2j,3j),E=(65000,0,0))

#Справочник проводов, тросов, изоляторов и опор воздушных ЛЭП
AC_150_24 = PVL.provod('AC-150/24', 0.198+0.000j, 17.1, 0.95)
S50=PVL.provod('С-50',3.75,9.1)
PS70D=PVL.izol('ПС-70Д',0.127)
PB110_1 = PVL.opora('ПБ110-1', [-2.0+14.5j, 3.5+14.5j, 2.0+17.5j], [0.0, 0.0, 0.0], 0.0+19.5j, 0.0+0.0j)

#Создание ветвей ВЛ от Sys1 до PS1 и PS2
sk1=PVL.sech('Сечение ВЛ 110 кВ от Sys1 до PS1 и PS2',50.0,0.05,1000.0)
pl1 = PVL.Line(sk1,'Линия Sys1-PS1',0.0,PB110_1.C1,AC_150_24,PS70D(8),RZT=0,fpr=5.5,q=(q1,q3))
t1 = PVL.Line(sk1,'Трос Sys1-PS1',0.0,PB110_1.T1,S50,0.0,RZT=2,fpr=5.5)
pl2 = PVL.Line(sk1,'Линия Sys1-PS2',30.0,PB110_1.C1,AC_150_24,PS70D(8),RZT=0,fpr=5.5,q=(q1,q4))
t2 = PVL.Line(sk1,'Трос Sys1-PS2',30.0,PB110_1.T1,S50,0.0,RZT=2,fpr=5.5)
mdl.ImportFromPVL(sk1)

#Создание ветвей ВЛ от Sys1 до PS1 и PS2
sk2=PVL.sech('Сечение ВЛ 110 кВ от Sys2 до PS1 и PS2',50.0,0.05,1000.0)
pl3 = PVL.Line(sk2,'Линия Sys2-PS1',0.0,PB110_1.C1,AC_150_24,PS70D(8),RZT=0,fpr=5.5,q=(q2,q3))
t3 = PVL.Line(sk2,'Трос Sys2-PS1',0.0,PB110_1.T1,S50,0.0,RZT=2,fpr=5.5)
pl4 = PVL.Line(sk2,'Линия Sys2-PS2',30.0,PB110_1.C1,AC_150_24,PS70D(8),RZT=0,fpr=5.5,q=(q2,q4))
t5 = PVL.Line(sk2,'Трос Sys2-PS2',30.0,PB110_1.T1,S50,0.0,RZT=2,fpr=5.5)
mdl.ImportFromPVL(sk2)

#Создание ветвей подстанций с трансформаторами с заземленными нейтралями
T1_PS1 = mrtkz.P(mdl,'T1 PS1',q3,q5,(52.9j,52.9j,47.61j),T=(115/10.5,11))
T1_PS2 = mrtkz.P(mdl,'T1 PS2',q4,q6,(52.9j,52.9j,47.61j),T=(115/10.5,11))

#Задание заземленного режима заземления нейтрали ВН трансформаторов PS1 и PS2
N1 = mrtkz.N(mdl,'Заземленная нейтраль T1 PS1',q5,'N0')
N2 = mrtkz.N(mdl,'Заземленная нейтраль T1 PS2',q6,'N0')

#Задание сопротивления нагрузки трансформаторов подстанций
Load1 = mrtkz.P(mdl,'Нагрузка T1 PS1',q5,0,(100,50j,50j))
Load2 = mrtkz.P(mdl,'Нагрузка T1 PS2',q6,0,(100,50j,50j))

#Создание однофазного КЗ
KZ1 = mrtkz.N(mdl,'Однофазное КЗ на шинах ВН PS1',q3,'A0')
#  Проверка на вырожденность
mdl.Test4Singularity()
#Формирование разреженной СЛАУ и расчет электрических параметров
mdl.Calc()
#Вывод результатов расчета для короткого замыкания
KZ1.res()
print()# Пустая строка

#Очистка модели от КЗ и обрывов за исключением типа 'N0'
mdl.ClearN()

# Пакетный расчет  КЗ других видов и в других точках

KZ1 = mrtkz.N(mdl,'Однофазное КЗ на шинах ВН PS1 через Rпер=2 Ом',q3,'A0r',r=2.0)
mdl.Calc()
KZ1.res()
print()# Пустая строка

mdl.ClearN()
KZ1 = mrtkz.N(mdl,'Двухфазное КЗ на шинах ВН PS1',q3,'BC')
mdl.Calc()
KZ1.res()
print()# Пустая строка

mdl.ClearN()
KZ1 = mrtkz.N(mdl,'Двухфазное КЗ на шинах ВН PS1 через Rпер=2 Ом',q3,'BCr',r=2.0)
mdl.Calc()
KZ1.res()
print()# Пустая строка

mdl.ClearN()
KZ1 = mrtkz.N(mdl,'Двухфазное КЗ на землю на шинах ВН PS1',q3,'BC0')
mdl.Calc()
KZ1.res()
print()# Пустая строка

mdl.ClearN()
KZ1 = mrtkz.N(mdl,'Двухфазное КЗ на шинах НН PS1',q5,'BC')
mdl.Calc()
KZ1.res()
print()# Пустая строка
T1_PS1.res1()
