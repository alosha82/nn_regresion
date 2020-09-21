Нужно два csv файла.Один для тренировки. Второй для тестирования.
Нужна не значительная коррекция кода под эти csv файлы.
Для запуска этой показательной версии нужны ещё три файла Weights-Adam-430--11.37813, Weights-Adamax-486--10.74177, Weights--Nadam-348--11.30503 формата hdf5.
Эти файлы и код связанный с ними не нужены для работы самой машины (эти файлы были созданы этой же программой).

Этот код нужен для предсказания(в нашем случае, потребления) путем использования алгоритма регрессии в виде нейронной сети.

После подправки кода под используемую базу данных код имеет данные ограничения:

Не понимает % или выражения из типа 70 - 80. Алгоритм воспримет это не как цифры.
Не относится к категории классификации.
Не работает с картинками.

Для запуска тренировки, отрисовки и функции подсчёта ошибки при условии что допустимая ошибка выше 90%, нужно активировать данные строки кода:

NN_model.fit(train, target, epochs=500, batch_size=32, validation_split = 0.2, callbacks=callbacks_list)

model = callbacks_list.pop()
NN_model = model.model
NN_model.compile(loss='mean_absolute_error', metrics=['mean_absolute_error'])
predictions = NN_model.predict(test)
result = compare_predicted_with_real(predictions, consumption)

plot4 = plt. figure(4)
plot4.suptitle('result', fontsize=16)
plt. plot(result)

print(under_90_percent(result))

Если вы желаете не использовать те три файла нужных для демонстрации показательных результатов, нужно деактивировать данные строки кода:

wights_file = 'Weights-Adam-430--11.37813.hdf5'  # choose the best checkpoint
NN_model.load_weights(wights_file) # load it
NN_model.compile(loss='mean_absolute_error', optimizer='Adam', metrics=['mean_absolute_error'])
predictions_adam = NN_model.predict(test)
result_adam = compare_predicted_with_real(predictions_adam, consumption)

wights_file = 'Weights--Nadam-348--11.30503.hdf5'  # choose the best checkpoint
NN_model.load_weights(wights_file) # load it
NN_model.compile(loss='mean_absolute_error', optimizer='Nadam', metrics=['mean_absolute_error'])
predictions_Nadam = NN_model.predict(test)
result_Nadam = compare_predicted_with_real(predictions_Nadam, consumption)

wights_file = 'Weights-Adamax-486--10.74177.hdf5'  # choose the best checkpoint
NN_model.load_weights(wights_file)  # load it
NN_model.compile(loss='mean_absolute_error', optimizer='Adamax', metrics=['mean_absolute_error'])
predictions_Adamax = NN_model.predict(test)
result_Adamax = compare_predicted_with_real(predictions_Adamax, consumption)

plot1 = plt. figure(1)
plot1.suptitle('Adam', fontsize=16)
plt. plot(result_adam)
plt.plot(x_values, y_values)
plot2 = plt. figure(2)
plot2.suptitle('NAdam', fontsize=16)
plt. plot(result_Nadam)
plt.plot(x_values, y_values)
plot3 = plt. figure(3)
plot3.suptitle('Adamax', fontsize=16)
plt. plot(result_Adamax)
plt.plot(x_values, y_values)

Если ваш файл тестирования содержит в себе Target столбец с реальными данными для проверки предсказанных, нужно активировать данные 
строки кода:

combined.reset_index(inplace=True)
combined.drop(['index', 'Target'], inplace=True, axis=1)


combined["Cloudiness"] = combined["Cloudiness"].fillna(0)
Данная строчка заполняет пустые места в столбце, нулями. Если не нужно то, деактивировать.