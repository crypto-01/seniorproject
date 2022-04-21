const menu = document.querySelector('#mobile-menu')
const menuLinks = document.querySelector('.navbar__menu')

//Display Mobile Menu
const mobileMenu = () => {
    menu.classList.toggle('is-active')
    menuLinks.classList.toggle('active')

}
//close menu after clicking
const close_after = () => {
  //menuLinks.classList.toggle('active')
    if (menuLinks.classList.contains("active")){
      menuLinks.classList.toggle('active')
      menu.classList.toggle('is-active')
    }

}

menu.addEventListener('click', mobileMenu)
menuLinks.addEventListener('click', close_after)
