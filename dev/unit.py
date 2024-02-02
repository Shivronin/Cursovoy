import unittest
import numpy as np
from main import process_text, DiffusionModel

class TestYourFunctions(unittest.TestCase):
    
    def test_process_text(self):
        # Проверка корректности обработки текста BERT
        text_batch = ["sample text 1", "sample text 2"]
        result = process_text(text_batch)
        self.assertTrue(np.array_equal(result.shape, (len(text_batch), 512)))  # Проверка размерности эмбеддингов

    def test_generate_images(self):
        # Подготовка данных для тестирования генерации изображений
        model = DiffusionModel(...)  # Замените многоточие на параметры вашей модели
        num_rows = 2
        num_cols = 2
        annotation = "Test annotation"
        ex_rate = 0.5
        size = 256

        # Тестирование генерации изображений
        generated_images = model.generate_images(num_images=num_rows * num_cols, annotation=annotation, ex_rate=ex_rate, size=size)
        
        # Проверка размерности сгенерированных изображений
        self.assertTrue(np.array_equal(generated_images.shape, (num_rows * num_cols, size, size, 3)))

if __name__ == '__main__':
    unittest.main()
