---
title: "Hello blogdown!"
subtitle: ''
author: Hana Lê 
excerpt: "Using blogdown to build a personal website"
date: 2023-01-28
# layout options: single, single-sidebar
layout: single
categories:
- R
- Blogdown
---



### Creating a Personal Website with Blogdown in R

I have used Rmarkdown and bookdown before but have never heard about blogdown until recently. I heard someone mention using blogdown to create a personal website and I was intrigued.  I was surprised to find that with a few hours of work, I was able to customize my site using this package. In this post, I'll share my experience and guide you through the process of using Blogdown to create your own website.

In this post, I'll share my experience of using blogdown to create a website and show you how you can do it too.

#### Prerequisites:

- Basic knowledge of R, RStudio and Rmarkdown
- Familiarity with GitHub

#### Step 1:Create a Repository on GitHub

1. Log in to your GitHub account and create a new repo.
2. Go to the repo's main page, click the green Code button and copy either the SSH or HTTPS link.

#### Step 2: Create a project in Rstudio

1. Click File > New Project > Version Control > Git.
2. Paste the copied link from GitHub.
3. Choose a location for the project on your workstation.
4. Click "Create Project."

#### Step 3: Create a Website

1. nstall the Blogdown package and a Hugo theme. Example:

```r
install.packages("blogdown")
blogdown::install_hugo()
library(blogdown)
new_site(theme = "hugo-apero/hugo/apero") 
# or "wowchemy/starter-academic 

```
 I initially started with the hugo-academic theme  but then found the hugo-apero theme fit me better.
 
 2. Preview the website locally using the command `blogdown::serve_site()` or the RStudio Addins menu.

Now it's time to see the sample. Click to “Show in new window” (to the right of the 🧹 icon) to preview it in a normal browser window. When you do that, you’ll be re-directed to the site’s main homepage.

#### Step 4: Personalize Content

1. Go to the content folder in your project to edit the markdown files.
2. Add front matter metadata like title, date, and author.
3. Add images to the static/img folder and reference them in your markdown files.
4. Customize the website's style with CSS by editing the CSS files in the theme folder.

#### Step 5: Publish website
1. Push the code to the GitHub repository.
When you’re happy with the content of your site, it’s time to publish it! To do this, you need to push your code to your GitHub repository.

```r
git add .
git commit -m "Your commit message"
git push origin master

```

2. Deploy the website using a service like Netlify by connecting it to the GitHub repo.
To make your website live, you need to deploy it. I hosted my website's source code on GitHub and created a new account on Netlify, which made it easy to deploy my site. Simply follow the instructions on the Netlify site to connect your GitHub repository and deploy your site.
And that's it! Your website should now be live and accessible to the public!

#### Conclusion

Creating a personal website using blogdown in R was easier and quicker than I thought. It was a fun and educational experience, and you'll now have a dynamic and professional-looking website to showcase your work and share your thoughts. Whether you're a data scientist, researcher, or just want to have a platform to share your interests, blogdown is a great option. Give it a try and see how you can create your own website today!


