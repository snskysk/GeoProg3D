PROMPT = """Think step by step to carry out the instruction.

The names of buildings and streets are enclosed in <>.

Instruction: A yellow building
Program:
ANSWER0=GetSimilarAreas(query='yellow building',area_filter_flag=False)
FINAL_RESULT=RESULT(var=ANSWER0)

Instruction: A rooftop on <Building A>
Program:
SEG0=GetLandmarkSeg(query='Building A')
ANSWER0=GetSimilarAreas(query='rooftop',area_filter_flag=True,area=SEG0)
FINAL_RESULT=RESULT(var=ANSWER0)

Instruction: Lawn area around the red signboard
Program:
AREAS0=GetSimilarAreas(query='red signboard',area_filter_flag=False)
SEG1=SegAround(seg=AREAS0)
ANSWER0=GetSimilarAreas(query='Lawn area',area_filter_flag=True,area=SEG1)
FINAL_RESULT=RESULT(var=ANSWER0)

Instruction: A yellow building between <Building A> and <Building B>
Program:
SEG0=GetLandmarkSeg(query='Building A')
SEG1=GetLandmarkSeg(query='Building B')
SEG2=SegBetween(seg_first=SEG0,seg_second=SEG1)
ANSWER0=GetSimilarAreas(query='yellow building',area_filter_flag=True,area=SEG2)
FINAL_RESULT=RESULT(var=ANSWER0)

Instruction: A red car in the area around <Street X>
Program:
SEG0=GetLandmarkSeg(query='Street X')
SEG1=SegAround(seg=SEG0)
ANSWER0=GetSimilarAreas(query='red car',area_filter_flag=True,area=SEG1)
FINAL_RESULT=RESULT(var=ANSWER0)

Instruction: The tall tower within 150 meters of <Building A>.
Program:
SEG0=GetLandmarkSeg(query='Building A')
SEG1=SegAround(seg=SEG0,size=150)
ANSWER0=GetSimilarAreas(query='tall tower',area_filter_flag=True,area=SEG1)
FINAL_RESULT=RESULT(var=ANSWER0)

Instruction: A telephone box northwest of <Building B>
Program:
SEG0=GetLandmarkSeg(query='Building B')
SEG1=SegDirection(seg=SEG0,query='northwest')
ANSWER0=GetSimilarAreas(query='telephone box',area_filter_flag=True,area=SEG1)
FINAL_RESULT=RESULT(var=ANSWER0)

Instruction: The skyscraper closest to <Building A>
Program:
CLOSEST_FROM0=GetLandmarkSeg(query='Building A')
AREAS0=GetSimilarAreas(query='skyscraper',area_filter_flag=False)
SEG0=GetCluster(seg=AREAS0,cluster_type='CLOSEST',from=CLOSEST_FROM0)
CLOSEST_TO0=SegAround(seg=SEG0)
ANSWER0=GetSimilarAreas(query='skyscraper',area_filter_flag=True,area=CLOSEST_TO0)
FINAL_RESULT=RESULT(var=ANSWER0)

Instruction: The bridge that is closest to the tall tower within 150 meters of <Building A>
Program:
SEG0=GetLandmarkSeg(query='Building A')
SEG1=SegAround(seg=SEG0,size=150)
AREAS0=GetSimilarAreas(query='tall tower',area_filter_flag=True,area=SEG1)
CLOSEST_FROM0=GetCluster(seg=AREAS0,cluster_type='BIGGEST')
AREAS1=GetSimilarAreas(query='bridge',area_filter_flag=False)
SEG2=GetCluster(seg=AREAS1,cluster_type='CLOSEST',from=CLOSEST_FROM0)
CLOSEST_TO0=SegAround(seg=SEG2)
ANSWER0=GetSimilarAreas(query='bridge',area_filter_flag=True,area=CLOSEST_TO0)
FINAL_RESULT=RESULT(var=ANSWER0)

Instruction: The building that is near the red sign between <Street A> and the sea
Program:
SEG0=GetLandmarkSeg(query='Street A')
AREAS0=GetSimilarAreas(query='sea',area_filter_flag=False)
SEG1=GetCluster(seg=AREAS0,cluster_type='BIGGEST')
SEG2=SegBetween(seg_first=SEG0,seg_second=SEG1)
AREAS1=GetSimilarAreas(query='red sign',area_filter_flag=True,area=SEG2)
SEG3=SegAround(seg=AREAS1)
ANSWER0=GetSimilarAreas(query='building',area_filter_flag=True,area=SEG3)
FINAL_RESULT=RESULT(var=ANSWER0)

Instruction: {instruction}
Program:

"""

