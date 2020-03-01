import pandas as pd
from ..predictor.predictor import predict


class TestMainPredictor:
    TEST_FILE_PATH = "x_test.csv"
    TEST_ANSWERS_FILE_PATH = "y_true.csv"

    def test_predict(self) -> None:
        data, answers = self._load_data_and_answers()
        print(answers)
        assert predict(data) == answers

    def _load_data_and_answers(self) -> Tuple[List[str], List[str]]:
        return pd.load_csv(self.TEST_FILE_PATH), pd.load_csv(self.TEST_ANSWERS_FILE_PATH)


if __name__ == '__main__':
    t = TestMainPredictor()
    t.test_predict()
