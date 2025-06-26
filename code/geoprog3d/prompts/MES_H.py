PROMPT = """Think step by step to carry out the instruction.

The names of buildings and streets are enclosed in <>.

Instruction: How tall is <Building A>?
Program:
SEG0=GetLandmarkSeg(query='Building A')
ANSWER0=GetAreaHeight(seg=SEG0)
FINAL_RESULT=RESULT(var=ANSWER0)

Instruction: How tall is the yellow building between <Building A> and <Building B>?
Program:
SEG0=GetLandmarkSeg(query='Building A')
SEG1=GetLandmarkSeg(query='Building B')
SEG2=SegBetween(seg_first=SEG0,seg_second=SEG1)
AREAS0=GetSimilarAreas(query='yellow building',area_filter_flag=True,area=SEG2)
SEG3=GetCluster(seg=AREAS0,cluster_type='BIGGEST')
ANSWER0=GetAreaHeight(seg=SEG3)
FINAL_RESULT=RESULT(var=ANSWER0)

Instruction: How tall is the tree around <Building A>?
Program:
SEG0=GetLandmarkSeg(query='Building A')
SEG1=SegAround(seg=SEG0)
AREAS0=GetSimilarAreas(query='tree',area_filter_flag=True,area=SEG1)
SEG2=GetCluster(seg=AREAS0,cluster_type='BIGGEST')
ANSWER0=GetAreaHeight(seg=SEG2)
FINAL_RESULT=RESULT(var=ANSWER0)

Instruction: How tall is the reddish tower near <Building A>?
Program:
SEG0=GetLandmarkSeg(query='Building A')
SEG1=SegAround(seg=SEG0)
AREAS0=GetSimilarAreas(query='reddish tower',area_filter_flag=True,area=SEG1)
SEG2=GetCluster(seg=AREAS0,cluster_type='BIGGEST')
ANSWER0=GetAreaHeight(seg=SEG2)
FINAL_RESULT=RESULT(var=ANSWER0)

Instruction: How tall is the skyscraper within 150m of <Building B>?
Program:
SEG0=GetLandmarkSeg(query='Building B')
SEG1=SegAround(seg=SEG0,size=150)
AREAS0=GetSimilarAreas(query='skyscraper',area_filter_flag=True,area=SEG1)
SEG2=GetCluster(seg=AREAS0,cluster_type='BIGGEST')
ANSWER0=GetAreaHeight(seg=SEG2)
FINAL_RESULT=RESULT(var=ANSWER0)

Instruction: How tall is the house to the east of <Building A>?
Program:
SEG0=GetLandmarkSeg(query='Building A')
SEG1=SegDirection(seg=SEG0,query='east')
AREAS0=GetSimilarAreas(query='house',area_filter_flag=True,area=SEG1)
SEG2=GetCluster(seg=AREAS0,cluster_type='BIGGEST')
ANSWER0=GetAreaHeight(seg=SEG2)
FINAL_RESULT=RESULT(var=ANSWER0)

Instruction: How tall is the building directly south of <building A>?
Program:
SEG0=GetLandmarkSeg(query='Building A')
SEG1=SegDirection(seg=SEG0,query='directly south')
AREAS0=GetSimilarAreas(query='building',area_filter_flag=True,area=SEG1)
SEG2=GetCluster(seg=AREAS0,cluster_type='BIGGEST')
ANSWER0=GetAreaHeight(seg=SEG2)
FINAL_RESULT=RESULT(var=ANSWER0)

Instruction: How tall is the building near the basketball court between <A Street> and the sea?
Program:
SEG0=GetLandmarkSeg(query='Building A')
AREAS0=GetSimilarAreas(query='sea',area_filter_flag=False)
SEG1=GetCluster(seg=AREAS0,cluster_type='BIGGEST')
SEG2=SegBetween(seg_first=SEG0,seg_second=SEG1)
AREAS0=GetSimilarAreas(query='basketball court',area_filter_flag=True,area=SEG2)
SEG3=GetCluster(seg=AREAS0,cluster_type='BIGGEST')
SEG4=SegAround(seg=SEG3)
AREAS1=GetSimilarAreas(query='building',area_filter_flag=True,area=SEG1)
SEG5=GetCluster(seg=AREAS1,cluster_type='BIGGEST')
ANSWER0=GetAreaHeight(seg=SEG5)
FINAL_RESULT=RESULT(var=ANSWER0)

Instruction: How tall is the house closest to <Building A>?
Program:
CLOSEST_FROM0=GetLandmarkSeg(query='Building A')
AREAS0=GetSimilarAreas(query='house',area_filter_flag=False)
SEG1=GetCluster(seg=AREAS0,cluster_type='CLOSEST',from=CLOSEST_FROM0)
CLOSEST_TO0=SegAround(seg=SEG1)
AREAS1=GetSimilarAreas(query='house',area_filter_flag=True,area=CLOSEST_TO0)
SEG2=GetCluster(seg=AREAS1,cluster_type='BIGGEST')
ANSWER0=GetAreaHeight(seg=SEG2)
FINAL_RESULT=RESULT(var=ANSWER0)


Instruction: How tall is the skyscraper closest to the tennis courts to the southeast of <Building A>?
Program:
SEG0=GetLandmarkSeg(query='Building A')
SEG1=SegDirection(seg=SEG0,query='southeast')
AREAS0=GetSimilarAreas(query='tennis courts',area_filter_flag=True,area=SEG1)
CLOSEST_FROM0=GetCluster(seg=AREAS0,cluster_type='BIGGEST')
AREAS1=GetSimilarAreas(query='skyscraper',area_filter_flag=False)
SEG2=GetCluster(seg=AREAS1,cluster_type='CLOSEST',from=CLOSEST_FROM0)
CLOSEST_TO0=SegAround(seg=SEG2)
AREAS2=GetSimilarAreas(query='skyscraper',area_filter_flag=True,area=CLOSEST_TO0)
SEG3=GetCluster(seg=AREAS2,cluster_type='BIGGEST')
ANSWER0=GetAreaHeight(seg=SEG3)
FINAL_RESULT=RESULT(var=ANSWER0)

Instruction: How tall is the tallest object within 100 meters of <Building A>?
Program:
SEG0=GetLandmarkSeg(query='Building A')
SEG1=SegAround(seg=SEG0,size=100)
ANSWER0=GetAreaHeight(seg=SEG1)
FINAL_RESULT=RESULT(var=ANSWER0)

Instruction: How tall is the tallest object to the south of <Building A>?
Program:
SEG0=GetLandmarkSeg(query='Building A')
SEG1=SegDirection(seg=SEG0,query='south')
ANSWER0=GetAreaHeight(seg=SEG1)
FINAL_RESULT=RESULT(var=ANSWER0)

Instruction: How tall is the tallest object between the soccer court to the northeast of <Building A> and <Building B>?
Program:
SEG0=GetLandmarkSeg(query='Building A')
SEG1=SegDirection(seg=SEG0,query='northeast')
AREAS0=GetSimilarAreas(query='soccer courts',area_filter_flag=True,area=SEG1)
SEG2=GetCluster(seg=AREAS0,cluster_type='BIGGEST')
SEG3=GetLandmarkSeg(query='Building B')
SEG4=SegBetween(seg_first=SEG2,seg_second=SEG3)
ANSWER0=GetAreaHeight(seg=SEG4)
FINAL_RESULT=RESULT(var=ANSWER0)

Instruction: How tall is the tallest object between the soccer court to the northeast of <Building A> and <Building A>?
Program:
SEG0=GetLandmarkSeg(query='Building A')
SEG1=SegDirection(seg=SEG0,query='northeast')
AREAS0=GetSimilarAreas(query='soccer courts',area_filter_flag=True,area=SEG1)
SEG2=GetCluster(seg=AREAS0,cluster_type='BIGGEST')
SEG3=GetLandmarkSeg(query='Building A')
SEG4=SegBetween(seg_first=SEG2,seg_second=SEG3)
ANSWER0=GetAreaHeight(seg=SEG4)
FINAL_RESULT=RESULT(var=ANSWER0)

Instruction: {instruction}
Program:

"""