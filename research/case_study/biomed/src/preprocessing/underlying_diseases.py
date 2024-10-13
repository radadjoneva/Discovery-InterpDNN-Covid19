# ruff: noqa: E402
# ruff: noqa: I001

import os
import sys

import pandas as pd

# Add biomed directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)


def extract_unique_diseases(df):
    # Split the list of diseases and flatten the list
    all_diseases = (
        df["Underlying diseases"]
        .dropna()
        .apply(lambda x: x.split(", ") if isinstance(x, str) else [x])
    )
    unique_diseases = set(disease for sublist in all_diseases for disease in sublist)
    return unique_diseases


def get_disease_frequencies(df):
    # Split and flatten the list of diseases
    all_diseases = (
        df["Underlying diseases"]
        .dropna()
        .apply(lambda x: x.split(", ") if isinstance(x, str) else [x])
    )
    disease_list = [disease for sublist in all_diseases for disease in sublist]

    # Count occurrences of each disease
    disease_counts = pd.Series(disease_list).value_counts()
    return disease_counts


def categorise_disease(disease):
    no_disease = {"No underlying disease", "No underlying diseases"}

    diabetes = {"Diabetes", "Hypertensiondiabetes", "Hypertension Diabetes", "Type 2 diabetes"}

    hypertension = {"Hypertension"}

    cardiovascular = {
        "Coronary atherosclerotic heart disease",
        "Heart disease",
        "Myocardial ischemia",
        "Arrhythmia",
        "Atrial fibrillation",
        "Myocardial infarction",
        "Angina pectoris",
        "Valvular heart disease",
        "Dilated cardiomyopathy",
        "Heart failure",
        "Aortic valve stenosis",
        "Coronary disease",
        "Percutaneous coronary intervention",
        "Rheumatic heart disease",
        "Hypertension heart disease",
        "Senile degenerative heart valve disease",
        "Vasovagal syncope",
        "Pericardial effusion",
        "Acute coronary syndrome",
        "Cardiac insufficiency",
        "Atrial flutter",
        "Chronic heart failure",
        "Hypertensive heart disease",
        "Premature ventricular contraction",
        "Heartdisease",
        "Myocardial bridge",
        "Apical ventricular aneurysm",
        "Severe heart failure",
        "Severe aortic regurgitation",
        "Hypertensionheart disease",
        "Hypertensioncoronary disease",
        "Sick sinus syndrome",
        "Senile degenerative valvular disease",
        "Tricuspid incompetence",
        "Mitral valve insufficiency",
        "Acute myocardial infarction",
        "Sinus rhythm",
        "Infiltrative cardiomyopathy",
        "Hyperthyroid heart disease",
        "Atherosclerosis",
        "Lower limb atherosclerosis",
        "Lower extremity arteriosclerotic occlusive disease",
        "Renal artery stenosis",
        "Aorta calcification",
        "Aortic dissection",
        "Atherosclerosis of aorta",
        "Vascular sclerosis",
        "Extensive atherosclerosis of arteries in both lower extremities",
        "Pulmonary artery widening with severe pulmonary hypertension",
    }

    cerebrovascular = {
        "Cerebral infarction",
        "Stroke",
        "Lacunar cerebral infarction",
        "Lacunar infarction",
        "Subdural hemorrhage",
        "Brain stem hemorrhage",
        "Cerebral hemorrhage",
        "Subdural effusion",
        "Multiple lacunar cerebral infarction",
        "Post-cerebral infarction",
        "Intracranial aneurysm",
        "Old cerebral infarction",
        "Cerebral atherosclerosis",
    }

    respiratory = {
        "Chronic obstructive pulmonary disease",
        "Asthma",
        "Pulmonary tuberculosis",
        "Bronchitis",
        "Emphysema",
        "Lung cancer",
        "Pulmonary artery widening with severe pulmonary hypertension",
        "Pulmonary heart disease",
        "Respiratory infections",
        "Bronchiectasis",
        "Chronic bronchitis",
        "Basic lung disease",
        "Bronchiectasia",
        "Pulmonary nodule",
        "Pleural effusion",
        "Chronic pharyngitis",
        "Old pulmonary tuberculosis",
        "Bronchial asthma",
        "Lung infections",
        "Respiratory infections",
        "Interstitial pneumonia",
        "Lung infection",
        "Fungal pneumonia",
        "Pulmonary embolism",
        "Bilateral bronchitis",
        "Pulmonary edema",
        "Chronic bronchitis of both lungs",
        "Chronic obstructive emphysema",
        "Interstitial lung disease",
        "Rhinitis",
        "Pulmonary encephalopathy",
        "COPD",
        "Obsolete tuberculosis",
        "Hypoxemia",
        "Pulmonary tuberculosis",
        "Right lung space occupying lesion",
        "Liver abscess",
        "Gangrenous cholecystitis",
        "Interstitial lesion of both lungs",
    }

    metabolic_liver = {
        "Fatty liver",
        "Hepatitis B",
        "Cirrhosis",
        "Hepatic cyst",
        "Hepatitis C",
        "Hepatic insufficiency",
        "Hepatic hemangioma",
        "Chronic hepatitis B",
        "Hepatic hemangioma",
        "Hepatitis B cirrhosis",
        "Hepatitis C cirrhosis",
        "Abnormal liver function",
        "Cirrhosis of liver",
    }

    metabolic_kidney = {
        "Renal failure",
        "Nephritis",
        "Kidney stones",
        "Renal cyst",
        "Nephrotic syndrome",
        "Chronic renal insufficiency",
        "Chronic renal failure",
        "Uremia",
        "Renal insufficiency",
        "Renal calculus",
        "Renal calculi",
        "Polycystic kidney",
        "Bladder stones",
        "Double kidney stones",
        "Diabetic nephropathy",
        "Left renal cyst",
        "Nephrosis",
        "Nephrotic syndrome",
        "Acute renal insufficiency",
    }

    metabolic_gastrointestinal = {
        "Gastritis",
        "Peptic ulcer",
        "Cholecystitis",
        "Gastric ulcer",
        "Pancreatitis",
        "Cholecystolithiasis",
        "Intestinal obstruction",
        "Chronic gastritis",
        "Erosive gastritis",
        "Chronic erosive gastritis",
        "Atrophic gastritis",
        "Gallstone",
        "Acute pancreatitis",
        "Gallbladder stones",
        "Duodenal ulcer",
        "Chronic cholecystitis",
        "Acute severe pancreatitis",
        "Erosive atrophic gastritis",
        "Intrahepatic bile duct stones",
        "Duodenal bulbar ulcer",
        "Chronic stomach disease",
        "Chronic superficial gastritis",
        "Space occupying lesion of pancreas",
        "Multiple gallstones",
    }

    metabolic_cardiovascular_risk = {
        "Obesity",
        "Hyperlipidemia",
        "Hypertriglyceridemia",
        "Hyperlipemia",
        "Hyperglycemia",
        "Hyperuricemia",
        "Lactic acidosis",
    }

    metabolic_other = {
        "Hyperthyroidism",
        "Hypothyroidism",
        "Hyperuricemia",
        "Hypoglycemia",
        "Hypoalbuminemia",
        "Hypokalemia",
        "Hyponatremia",
        "Hypopotassaemia",
        "Lactic acidosis",
        "Obstructive jaundice",
    }

    cancer = {
        "Breast cancer",
        "Colon cancer",
        "Lung cancer",
        "Prostate cancer",
        "Ovarian cancer",
        "Cervical cancer",
        "Gastric cancer",
        "Liver cancer",
        "Leukemia",
        "Lymphoma",
        "Thyroid cancer",
        "Acute myeloid leukemia",
        "Malignant tumor",
        "Acute lymphoblastic leukemia",
        "Multiple myeloma",
        "Chronic lymphocytic leukemia",
        "Lung metastases",
        "Squamous cell carcinoma of the lower lobe of the right lung",
        "Non Hodgkin's lymphoma",
        "Gastric cancer",
        "Prostate tumor",
        "Lung tumor",
        "Gynecologic tumor",
        "Type B lymphocytic leukemia",
        "Prostate cancer with bone metastasis",
        "Pancreatic cancer with metastasis",
    }
    immune_system = {
        "Rheumatoid arthritis",
        "Lupus",
        "Autoimmune hemolytic anemia",
        "Ankylosing spondylitis",
        "Hashimoto's thyroiditis",
        "Gout",
        "Systemic lupus erythematosus",
        "Hashimoto thyroiditis",
        "ANCA-associated vasculitis",
        "Idiopathic thrombocytopenic purpura",
        "ANCA-associated glomerulonephritis",
        "Autoimmune hemolytic anemia",
        "Severe immunodeficiency",
        "Rheumatism",
        "Myasthenia gravis",
        "Severe aplastic anemia",
        "Malignant tumor of gastric angle",
    }

    if disease in no_disease:
        return "No underlying disease"
    elif disease in cardiovascular:
        return "Cardiovascular"
    elif disease in cerebrovascular:
        return "Cerebrovascular"
    elif disease in respiratory:
        return "Respiratory"
    elif disease in metabolic_liver:
        return "Metabolic liver"
    elif disease in metabolic_kidney:
        return "Metabolic kidney"
    elif disease in metabolic_gastrointestinal:
        return "Metabolic gastrointestinal"
    elif disease in metabolic_cardiovascular_risk:
        return "Metabolic and cardiovascular risk"
    elif disease in metabolic_other:
        return "Metabolic other"
    elif disease in cancer:
        return "Cancer"
    elif disease in immune_system:
        return "Immune system"
    elif disease in diabetes:
        return "Diabetes"
    elif disease in hypertension:
        return "Hypertension"
    else:
        return "Other"


if __name__ == "__main__":
    file_path = "research/case_study/biomed/datasets/iCTCF/cleaned_cf_data.csv"
    df = pd.read_csv(file_path)

    unique_diseases = extract_unique_diseases(df)
    print(f"Number of unique diseases: {len(unique_diseases)}")

    # Get disease frequencies
    disease_counts = get_disease_frequencies(df)

    print(disease_counts)
