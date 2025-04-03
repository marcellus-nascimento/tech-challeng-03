import pandas as pd


def summary(df):
    """Retorna um resumo estatístico do dataframe."""
    summary_df = pd.DataFrame(df.dtypes, columns=['dtypes'])
    summary_df['missing#'] = df.isna().sum()
    summary_df['missing%'] = (df.isna().sum()) / len(df)
    summary_df['unique'] = df.nunique().values
    summary_df['count'] = df.count().values
    return summary_df


def clean_and_combine(df1, df2, df3):
    """Realiza os tratamentos necessários e combina os dataframes."""

    # Normalizar o rótulo em binário e renomear
    df1['Diabetes_012'] = df1['Diabetes_012'].replace({1: 1, 2: 1, 0: 0})
    df1.rename(columns={'Diabetes_012': 'Diabetes_binary'}, inplace=True)

    # Comparação entre df2 e df3
    if df2.equals(df3):
        print("Os DataFrames df2 e df3 são idênticos.")
    else:
        print("Os DataFrames df2 e df3 são diferentes.")

    # Combina os dataframes
    df = pd.concat([df1, df2, df3], ignore_index=True)

    # Tratamento de colunas
    df.rename(columns=lambda x: x.strip(), inplace=True)
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    df.drop(["nodocbccost","anyhealthcare", "menthlth", "physhlth","education","income", "diffwalk"], axis=1, inplace=True)

    return df
