# Обновите определение класса LogRegressor
# Ваш код здесь
class LogRegressor():
    
    """Конструктор
    
    Параметры
    ----------
    tags : list of string, default=top_tags    #список тегов
    """
    def __init__(self, tags=top_tags):      
        # словарь который содержит мапинг слов предложений и тегов в индексы (для экономии памяти)
        # пример: self._vocab['exception'] = 17 означает что у слова exception индекс равен 17
        self._vocab = {}
        
        # параметры модели: веса
        # для каждого класса/тега нам необходимо хранить собственный вектор весов
        # по умолчанию у нас все веса будут равны нулю
        # мы заранее не знаем сколько весов нам понадобится
        # поэтому для каждого класса мы сосздаем словарь изменяемого размера со значением по умолчанию 0
        # пример: self._w['java'][self._vocab['exception']]  содержит вес для слова exception тега java
        self._w = dict([(t, defaultdict(int)) for t in tags])
        
        # параметры модели: смещения или вес w_0
        self._b = dict([(t, 0) for t in tags])
        
        self._tags = set(tags)
    
    """Один прогон по датасету
    
    Параметры
    ----------
    fname : string, default=DS_FILE_NAME
        имя файла с данными
        
    top_n_train : int
        первые top_n_train строк будут использоваться для обучения, остальные для тестирования
        
    total : int, default=10000000
        информация о количестве строк в файле для вывода прогресс бара
    
    learning_rate : float, default=0.1
        скорость обучения для градиентного спуска
        
    tolerance : float, default=1e-16
        используем для ограничения значений аргумента логарифмов
    """
    def iterate_file(self, 
                     fname=DS_FILE_NAME, 
                     top_n_train=100000, 
                     total=125000,
                     learning_rate=0.1,
                     tolerance=1e-15,
                     lm = 0.0002,
                     gamma=0.1):   #regularisation parameter
        
        self._loss = []
        self._acc = []    #для подсчёта средней точности работы
        n = 0
        word_set = set()
        
        # откроем файл
        with open(fname, 'r') as f:            
            # прогуляемся по строкам файла
            for line in tqdm_notebook(f, total=total, mininterval=1):
                pair = line.strip().split('\t')
                # print(pair)
                if len(pair) != 2:
                    continue                
                sentence, tags = pair
                # слова вопроса, это как раз признаки x
                sentence = sentence.split(' ')
                # теги вопроса, это y
                tags = set(tags.split(' '))
                print ("Original question tags: ", tags)
                
                # значение функции потерь для текущего примера
                sample_loss = 0
                
                p_tags = set()   #предсказанные теги вопроса
                word_set.clear()

                # прокидываем градиенты для каждого тега
                for tag in self._tags:
                    # целевая переменная равна 1 если текущий тег есть у текущего примера
                    y = int(tag in tags)
                    
                    #набираем множество реальных тегов для тестового вопроса
                   # if n >= top_n_train and y :
                    #    r_tags = r_tags.append(tag) 
                    
                    # расчитываем значение линейной комбинации весов и признаков объекта
                    z = 0
                    rw = 0
                    dLdw = 0
                       
                    for word in sentence:
                        # если в режиме тестирования появляется слово которого нет в словаре, то мы его игнорируем
                        if n >= top_n_train and word not in self._vocab:
                            continue
                        if word not in self._vocab:
                            self._vocab[word] = len(self._vocab)
                        if word not in word_set:
                            word_set.add(word)
                            #если встретили слово в первый раз, доб. регресс член в производную
                            w_ik = self._w[tag][self._vocab[word]]
                            dLdw += lm*(2*gamma*w_ik \
                                  + (1 - gamma)*np.sign(w_ik))
                            
                        z += self._w[tag][self._vocab[word]] + self._b[tag]
                        rw += pow(self._w[tag][self._vocab[word]], 2)
    
                    # вычисляем вероятность наличия тега sigma
                    lim = - np.log(tolerance/(1 - tolerance))
                    if z < -lim:
                        sigma = tolerance
                    elif z > lim:
                        sigma = 1 - tolerance
                    else:
                        sigma = 1/ (1 + np.exp(-z))
    
                    
                    # обновляем значение функции потерь для текущего примера
                    sample_loss += y*np.log(sigma) + (1 - y)*np.log(1 - sigma) 
                #+ lm*rw/2

                    if n < top_n_train: 
                        dLdw += y - sigma
                        for word in sentence: 
                            #dLdw += lm*self._w[tag][self._vocab[word]]
                            self._w[tag][self._vocab[word]] -= -learning_rate*dLdw
                        self._b[tag] -= -learning_rate*dLdw
                        
                    #если мы уже в тестировочной части
                    else:
                        #считаем вероятность для данного примера содержать текущий тег (как?)
                        #если такая веротность > 0.9, добавляем этот тег в множетсов тегов примера
                        if (sigma > 0.9):
                            p_tags.add(tag)
                        #смотрим множетсво реальных тегов примера (как?)
                        #смотрим множетсво предсказанных тегов примера
                        #считаем коэффециент Жаккара (он же acc, точность) для них
                    
                #на выходе из цикла по тегам у нас заполнено два множетсва, r_tags = и p_tags  
                #и мы можем посчитать коэффт Жаккара (как точность (accuracy) модели:
                
                if n >= top_n_train:
                    acc = JacCoeff(tags, p_tags)
                    self._acc.append(acc)
                
                n += 1
                        
                self._loss.append(sample_loss)
        #прошлись по всему файлу, теперь у нас есть массив точностей по всем примерам и мы
        #можем вернуть пользователю среднюю точность
        return np.mean(self._acc)
