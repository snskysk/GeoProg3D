const modelViewerGallary = document.querySelector('#modelViewerGallary');
modelViewerGallary.isTextured = true;

// Initialize the selection panel images
$('#gallerySelectionPanel .selectable-image').each((i, img) => {
    img.src = `static/gallery/${img.getAttribute('name')}/image.jpg`;
})

// Add event listener to the selection panel
const gallerySelectionPanel = document.getElementById('gallerySelectionPanel');
gallerySelectionPanel.addEventListener('click', async function(event) {
    const img = event.target.closest('.selectable-image'); 
    
    if (!img || img.classList.contains('selected')) 
        return;
   
    gallerySelectionPanel.querySelectorAll('.selectable-image').forEach(function(image) {
        image.classList.remove('selected');
    });
    img.classList.add('selected');

    const name = img.getAttribute('name');

    modelViewerGallary.src = `static/gallery/${name}/mesh.glb`;
    modelViewerGallary.texturePath = `static/gallery/${name}/image.jpg`;
    modelViewerGallary.resetView();
    modelViewerGallary.showPoster();
});

// Set the toggle buttons
toggleGalleryLeftButton = document.querySelector('#toggleTexturedGallery .toggle-left');
toggleGalleryRightButton = document.querySelector('#toggleTexturedGallery .toggle-right');

toggleGalleryLeftButton.addEventListener('click', function() {
    toggleGalleryLeftButton.classList.add('active');
    toggleGalleryRightButton.classList.remove('active');
    modelViewerGallary.setTextured(false);
});

toggleGalleryRightButton.addEventListener('click', function() {
    toggleGalleryRightButton.classList.add('active');
    toggleGalleryLeftButton.classList.remove('active');
    modelViewerGallary.setTextured(true);
});

// Initialize the model viewer with selected model
$(document).ready(() => {
    const name = document.querySelector('#gallerySelectionPanel .selectable-image.selected').getAttribute('name');

    modelViewerGallary.src = `static/gallery/${name}/mesh.glb`;
    modelViewerGallary.texturePath = `static/gallery/${name}/image.jpg`;
    
    modelViewerGallary.resetView();

    modelViewerGallary.showPoster();  
});