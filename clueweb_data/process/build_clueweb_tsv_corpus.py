import sys

# adding path
sys.path.insert(0, '/home/jingyuah/ClueWeb22')
from ClueWeb22Api import ClueWeb22Api

import json
from tqdm import tqdm
import re

# set this to save space...
MAX_WORDS = 128

print(sys.argv[0])
print(sys.argv[1])

subfolder_id = str(sys.argv[1]).zfill(2)

# path_out = "/data/group_data/cx_group/clueweb22b-corpus/corpus"
path_out = "/home/jingyuah"
path_clueweb = "/data/datasets/clueweb22/ClueWeb22_B"

clueweb2int = {}
start_int = 0
print(f"en00{subfolder_id}")
print("#################")


# run like:
# python build.py 0
# will process subfolder en0000 of the dataset
# ..
# python build.py 46
# will process subfolder en0046 of the dataset

with open(f'{path_out}/full_corpus_en00{subfolder_id}.tsv', 'w') as out: 
        for jsongz_id in tqdm(range(0, 100), desc="JsonGZ-ID"):
            for doc_id in range(0, 25000):

                jjsongz_id = str(jsongz_id).zfill(2)
                ddoc_id = str(doc_id).zfill(5)

                cweb_doc_id = f"clueweb22-en00{subfolder_id}-{jjsongz_id}-{ddoc_id}"
                clueweb_api = ClueWeb22Api(cweb_doc_id, path_clueweb)
                # print("cweb_doc_id", cweb_doc_id)

                try:
                    clean_txt = eval(clueweb_api.get_clean_text())
                    cweb_id = clean_txt["ClueWeb22-ID"]
                    content = clean_txt["Clean-Text"]
                    title = content.split('\n')[0].replace("\n", "").replace("\t", "").replace("\r", "").replace("\'", "").replace("\"", "").strip()
                    content = content.replace("\n", "").replace("\t", "").replace("\r", "").replace("\'", "").replace("\"", "").strip()
                    #content = re.sub(r'[\n\t\r\'"]', '', content).strip()
                    
                except Exception as e:
                    # not all jsongz ids have same number of docs (max is ~25k). there is a txt with a count but i just let it raise an exception
                    continue
                
                # write to tsv format
                content = " ".join(content.split(" ")[:MAX_WORDS])
                print("id: ", cweb_doc_id)
                print("content: ", content)
                exit()
                out.write(cweb_id + "\t" + title + "\t" + content + "\n")
                

