import sys

# adding path
sys.path.insert(0, '/home/jingyuah/ClueWeb22') # CHANGE: your ClueWeb22 source
from ClueWeb22Api import ClueWeb22Api

import json
from tqdm import tqdm
import re

# set this to save space...
MAX_WORDS = 128

print(sys.argv[0])
print(sys.argv[1])

subfolder_id = str(sys.argv[1]).zfill(2)

path_out = "/home/jingyuah" # CHANGE: output_path
path_clueweb = "/data/datasets/clueweb22/ClueWeb22_B" # CHANGE: path_to_clueweb_corpus

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

with open(f'{path_out}/full_corpus_en00{subfolder_id}.tsv', 'w') as out: # CHANGE: output_file
        for jsongz_id in tqdm(range(0, 100), desc="JsonGZ-ID"):
            for doc_id in range(0, 25000):

                jjsongz_id = str(jsongz_id).zfill(2)
                ddoc_id = str(doc_id).zfill(5)

                cweb_doc_id = f"clueweb22-en00{subfolder_id}-{jjsongz_id}-{ddoc_id}"
                clueweb_api = ClueWeb22Api(cweb_doc_id, path_clueweb)
                # print("cweb_doc_id", cweb_doc_id)
           
                try:
                    inlink = eval(clueweb_api.get_inlinks()) # the api used for inlink file from the ClueWeb repo
                    cweb_id = inlink["ClueWeb22-ID"]
                    anchors = inlink["anchors"] # list of anchors
                    # The i'th anchor record is a list of five values: [url, urlhash, anchor text, ?, language]
                    # example anchor: ['https://www.amazon.com/Transpersonal-Psychology/b?node=10166948011', \
                        # 34F41FC98C9F2271C5B1E53C365AD1A1', \
                        # 'price$53.88\n\n\n\n                    $74.00\n                \n\n\n\n\n            Psychoanalytic Object Relations Therapy\n        \n\n\n\n2', \
                        # '0', 'en', 'clueweb22-en0803-08-03107']
                    
                    # CHANGE: process the anchor information as the way you want
                    for anchor in anchors: 
                        anchor_text = anchor[2].replace("\n", "").replace("\t", "")
                    
                except Exception as e:
                    # not all jsongz ids have same number of docs (max is ~25k). there is a txt with a count but i just let it raise an exception
                    continue
                
                
                
                # CHANGE: write the info to the output file
                
                # write to tsv format
                # content = " ".join(content.split(" ")[:MAX_WORDS])
                # out.write(cweb_id + "\t" + title + "\t" + content + "\n")

