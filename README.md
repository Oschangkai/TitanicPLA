# Titanic PLA

**所有詳細資料都在 `titanic.py` 中的註解**

## 檔案們的介紹
- `result.0` 是保留 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch' 欄位的資料當 training，另外新增的 'Family' 欄位
- `result.1` 是保留 'Pclass', 'Sex', 'Age' 欄位的資料當 training
- `result.2` 是保留 'Sex', 'Age' 欄位的資料當 training
- 檔名包含 `.dropna` 的是丟棄 Sex 包含 N/A 的資料
- 檔名包含 `.out` 是 console 印出來的資訊，包含了每一次 iteration 的錯誤率、最後的 W, err_rate 和與範例輸出資料的比對程度
- 檔名包含 `.csv` 是預測資料