import pandas as pd
import pdfplumber

csv_path = "../data/raw_data/Project Management (1).csv"
pdf_path = "../data/raw_data/final_project_guideline.pdf"
plain_path = "../data/raw_data/emails.csv"


def explore_csv(csv_path):
  df = pd.read_csv(csv_path)
  print("=====First 2 rows=====\n", df.head(2))
  print("\n=====Size=====\n", df.shape)
  print("\n=====Missing Values=====\n", df.isnull().sum())

explore_csv(csv_path)


def extract_section(text, start_txt, end_txt):
  start = text.find(start_txt)
  end = text.find(end_txt)
  between = text[start:end]
  elem = between[between.find(":")+2:]
  return elem


def explore_pdf(pdf_path):
  full_text = []
  with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
      page_text = page.extract_text()
      if page_text:
        page_text = " ".join(page_text.split()) 
        full_text.append(page_text)
  full_text_string = " ".join(full_text)
  print("=====First 200 words=====\n", full_text[:200])

  print("=====Fetching Fields=====\n")
  content_list = []
  section_header = ['Deadline','Submission Meterials','Required Elements']
  text = full_text_string

  #due date
  start_txt = "Time:"
  end_txt = ": Final projects due"
  content_list.append(extract_section(text, start_txt, end_txt))

  #submission Materials
  start_txt = "Materials to turn in:"
  end_txt = "The project report must have the following elements:"
  content_list.append(extract_section(text, start_txt, end_txt))

  #Required Fields
  start_txt = "The project report must have the following elements:"
  end_txt = "*"
  content_list.append(extract_section(text, start_txt, end_txt))

  df = pd.DataFrame([content_list], columns = section_header)
  print(df.head(2))


  pdf_save_path = "../data/processed_data/pdf_csv.csv"
  df.to_csv(pdf_save_path, index=False)
  print("\n=====Save result at=====\n", pdf_save_path)


explore_pdf(pdf_path)


df = pd.read_csv(plain_path)
df = df.head(100) 
end_path = "../data/processed_data/emails_sample.csv" #test only with the first 100 email samples, saved in a separte csv file
df.to_csv(end_path, index = False)

def explore_text(end_path):
  df = pd.read_csv(end_path)

  id_list = []
  date_list = []
  from_list = []
  to_list = []
  subject_list = []
  body_list = []

  for row in df["message"]:

    header = row.split("\n\n",1)[0]
    body = row.split("\n\n",1)[1]
    body_list.append(body.strip())


    for line in header.split("\n"):
      ##message id
      if line.startswith("Message-ID:"):
        field_val = line.replace("Message-ID:", "").strip()
        id_list.append(field_val)

      ##date
      if line.startswith("Date:"):
        field_val = line.replace("Date:", "").strip()
        date_list.append(field_val)

      ##From
      if line.startswith("From:"):
        field_val = line.replace("From:", "").strip()
        from_list.append(field_val)

      ##To
      if line.startswith("To:"):
        field_val = line.replace("To:", "").strip()
        to_list.append(field_val)

      ##Subject
      if line.startswith("Subject:"):
        field_val = line.replace("Subject:", "").strip()
        subject_list.append(field_val)
    
  df = pd.DataFrame({"id":id_list, 
                     "date":date_list,
                     "from":from_list,
                     "to":to_list, 
                     "subject":subject_list,
                     "body":body_list})
  
  print("=====Parsed Data=====\n", df.head(2))
  print("\n=====Size=====\n", df.shape)
  print("\n=====Missing Values=====\n", df.isnull().sum())

  
  plain_save_path = "../data/processed_data/txt_csv.csv"
  df.to_csv(plain_save_path, index=False)
  print("\n=====Save result at=====\n", plain_save_path)
  

explore_text(end_path)
