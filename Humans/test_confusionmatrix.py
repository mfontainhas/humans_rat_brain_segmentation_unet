def model_print_predictions(model,input_attributes,output_attributes):
    predictions = model.predict(input_attributes)
    # arredondar para 0 ou 1 pois pretende-se um output binário
    LP=[]
    for prev in predictions:
        LP.append(round(prev[0]))
    #LP = [round(prev[0]) for prev in predictions]
    for i in range(len(output_attributes)):
        print(" Class:",output_attributes[i]," previsão:",LP[i])
        if i>10: break
    TP=0
    TN=0
    FP=0
    FN=0
    for i in range(len(LP)):
        if output_attributes[i]==1 and LP[i] == 1: TP+=1
        elif output_attributes[i]==0 and LP[i] == 0: TN+=1
        elif output_attributes[i]==0 and LP[i] == 1: FP+=1
        elif output_attributes[i]==1 and LP[i] == 0: FN+=1
    print("TP: "+TP+";  " +"TN: "+TN+";  "+"FP: " +FP+";  "+"FN: "+FN+"; ")
    print("Accuracy: " + ((TP+TN)/(TP+TN+FP+FN)))