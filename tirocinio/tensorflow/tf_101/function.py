def get_clothing_item(name_item):
  index_item = None
  
  for i in range(0,9):
    if name_item == class_names[i]:
      index_item = i
      break
  
  if(index_coat is None):
    print("errore, capo d'abbigliamento non presente! MA QUE CERCHI")
    return

  index = rand.randint(1,len(test_images)-1)
  while True:  
    if np.argmax(predictions[index]) == index_item:
      break 
    index =  rand.randint(1,len(test_labels)-1)
    print("sto ancora cercando! Dammi tempo!\n")
    
  plt.figure()
  plt.imshow(test_images[index])
  plt.colorbar()
  plt.grid(False)
  plt.show()





