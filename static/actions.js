{/* <li class="nav-item"><a class="nav-link" href="/braintumor">Brain Tumor</a></li>
              <!-- <li class="nav-item"><a class="nav-link" href="/breastcancer">Breast Cancer</a></li> -->
              <li class="nav-item"><a class="nav-link" href="/alzheimer">Alzheimer</a></li>
              <li class="nav-item"><a class="nav-link" href="/diabetes">Diabetes</a></li>
              <li class="nav-item"><a class="nav-link" href="/pneumonia">Pneumonia</a></li>
              <li class="nav-item"><a class="nav-link" href="/heartdisease">Heart Disease</a></li>
              <li class="nav-item"><a class="nav-link active" href="/symptoms-based">Symptoms Based</a></li> */}
// html = document.getElementsByTagName('html')[0]
// html.setAttribute('style', 'overflow: hidden;')
options = ['Brain Tumor', 'Alzheimer', 'Diabetes', 'Pneumonia', 'Heart Disease', "Many In One", 'Symptoms Based'];
navbarSC = document.getElementById('nb-s-c-ul');
description = { 'Brain Tumor': "A Densenet121 model ", 'Alzheimer': "", "Diabetes": "", "Pneumonia": "", "Heart Disease": "", "Symptoms Based": "" }
// "".lore

// type = "{{ diag_type }}"
// console.log(type)

options.forEach(el => {
  if(el!="Many In One"){
    element = document.createElement('li')
    element.innerHTML = `<a id="nb-${el.replace(" ", "").toLowerCase()}"  class="nav-link" href="/predict/${el.replace(" ", "").toLowerCase()}" style="color:white; font-size=10rem">${el}</a>`;
    navbarSC.appendChild(element);
  }
});

diagcards = document.getElementById('diag-cards');
// div = document.createElement("div")
index = 0

console.log(active)

nb_a = document.getElementById(`nb-${active}`)
if (nb_a != undefined) nb_a.style.fontWeight = 'bold'

// "".replace
options.forEach(el => {

  // console.log(el.split(" ")[0])
  element = document.createElement('form')
  element.setAttribute('style', 'width: auto; margin: 0ch 0 4ch 0px')
  element.setAttribute('action', `/predict/${el.replaceAll(" ", "")}`)
  element.setAttribute('method', `get`)
  element.setAttribute('enctype',"multipart/form-data")
  if (el == "Many In One") {
    element.setAttribute('method', `post`)
    console.log(el.replaceAll(" ", "").toLowerCase());
    element.innerHTML = `<div class="card" style="width:42rem; height:278.2px">
    <div style="background-image: url('./static/${el.replaceAll(" ", "").toLowerCase()}_img.jpg'); background-size: 100% 100%; height: 77%; width: 100%; display: flex; justify-content: center; align-items: center;">
    <a id='upload_doc' class="btn btn-primary">
  <i class="fas fa-upload upload-icon"></i>
    Upload Report</a> 
    </div>
  <!--<img src="./static/${el.replaceAll(" ", "").toLowerCase()}_img.jpg" style="height:77%;" class="card-img-top" alt="...">-->
  <div class="card-body">
    <h5 class="card-title">${el}</h5>
    <!--<p class="card-text">Some quick example text to build on the card title and make up the bulk of the card's content.</p>
   <a class="card-header bg-transparent border-primary" style="color: red; margin: none; text-decoration: none; " href="#">Learn</a>-->
    <input type="file" id="file_input"  name="file" accept="*/*" style="display: none;"/>
    <button type="submit" id="manyinone_sub" style="display:none;"></button>
  </div>
</div>`
  } else {
    //     element.innerHTML = `<div class="card border-primary mb-3" style="max-width: 18rem;">
    //   <a class="card-header bg-transparent border-primary" style="color: red; text-decoration: none; " href="#">Learn</a>
    //   <div class="card-body ">
    //     <h5 class="card-title">${el}</h5>
    //     <p class="card-text">Some quick example text to build on the card title and make up the bulk of the card's content.</p>
    //   </div>
    //   <div class = "predict-footer"><div class="card-footer bg-transparent" style="border:none;"><button type="submit" class="btn btn-outline-primary">Predict</button></div></div>
    // </div>`
    element.innerHTML = `<div class="card" style="width: 18rem;">
  <img src="./static/${el.replaceAll(" ", "").toLowerCase()}_img.jpg" style="height:175px;" class="card-img-top" alt="...">
  <div class="card-body">
    <h5 class="card-title">${el}</h5>
    <!--<p class="card-text">Some quick example text to build on the card title and make up the bulk of the card's content.</p>
   <a class="card-header bg-transparent border-primary" style="color: red; margin: none; text-decoration: none; " href="#">Learn</a>-->
    <button type="submit" class="btn btn-primary">Try Prediction Model</button>
  </div>
</div>`
  }
  diagcards.appendChild(element)
})

upload_doc = document.getElementById('upload_doc');
file_inp = document.getElementById('file_input');

upload_doc.addEventListener('click', ()=>file_inp.click());
file_inp.addEventListener('change', ()=>{
  console.log(file_inp);
  manyinone_sub = document.getElementById('manyinone_sub');
  manyinone_sub.click();
})

console.log(active)

nb_a = document.getElementById(`nb-${active}`)
nb_a.style.fontWeight = 'bold'



