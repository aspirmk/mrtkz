#Импортирование модуля расчета ТКЗ (mrtkz3.py),
#который должен находиться в той же папке, где и настоящий файл
import mrtkz3 as mrtkz
#Создание расчетной модели
mdl=mrtkz.Model()

#Создание узлов
q1 = mrtkz.Q(mdl,'Sys1')
q2 = mrtkz.Q(mdl,'Sys2')
q3 = mrtkz.Q(mdl,'PS1')
q4 = mrtkz.Q(mdl,'PS2')

#Создание ветвей энергосистем
Sys1 = mrtkz.P(mdl,'Sys1',0,q1,(2j,2j,3j),E=(65000,0,0))
Sys2 = mrtkz.P(mdl,'Sys2',0,q2,(2j,2j,3j),E=(65000,0,0))

#Создание ветвей Воздушных линий
Line1 = mrtkz.P(mdl,'Sys1-PS1',q1,q3,(10j,10j,30j))
Line2 = mrtkz.P(mdl,'Sys1-PS2',q1,q4,(10j,10j,30j))
Line3 = mrtkz.P(mdl,'Sys2-PS1',q2,q3,(10j,10j,30j))
Line4 = mrtkz.P(mdl,'Sys2-PS2',q2,q4,(10j,10j,30j))
#Создание взаимоиндукций нулевой последовательности между Воздушными линиями
M12 = mrtkz.M(mdl,'L1-L2',Line1,Line2,15j,15j)
M34 = mrtkz.M(mdl,'L3-L4',Line3,Line4,15j,15j)

#Создание ветвей подстанций с трансформаторами с заземленными нейтралями
PS1 = mrtkz.P(mdl,'PS1',0,q3,(500,200j,30j))
PS2 = mrtkz.P(mdl,'PS2',0,q4,(500,200j,30j))

#Создание КЗ
KZ1 = mrtkz.N(mdl,'KZ',q3,'A0')
#KZ1 = mrtkz.N(mdl,'KZ',q3,'BC')
#KZ1 = mrtkz.N(mdl,'KZ',q3,'BC0')
#KZ1 = mrtkz.N(mdl,'KZ',q3,'A0r',r=2.0)

#  Проверка на вырожденность
mdl.Test4Singularity()

#Формирование разреженной СЛАУ и расчет электрических параметров
mdl.Calc()

#Вывод результатов расчета для короткого замыкания
KZ1.res()
