---
layout: distill
title: Beauty Inc Dashboard
description:
img: assets/img/beauty_inc_dashboard/dashboard_thumbnail_square.png
importance: 2
related_publications: false
toc : true
enable_math: false
---
# Project description

## Business problem

A beauty product e-store aims to achieve the following targets:
- average annual profit margin of 15% across all product groups ,
- annual overall sales growth of 20% with a higher target of 30% for the corporate segment, and
- at least $400K in annual sales for each market

The objective is to analyse sales volumes and profitability across various product groups and markets, focussing on key performance indicators (KPIs) critical to the e-store's success.

## Data description

The data is taken from the following source: [zoomchart-challenge](https://zoomcharts.com/en/microsoft-power-bi-custom-visuals/challenges/fp20-analytics-september-2024)


{% details Brief description of columns included in the dataset %}
    
| Row ID | Unique ID of the row |
| Order ID | ID of the order (one order can have multiple items) |
| Order Date | Date when order was made |
| Customer ID | ID of the customer (one customer can make multiple orders) |
| Segment | Customer segment |
| City | Order city |
| State | Order state (where applicable) |
| Country | Order country |
| Country latitude | Geographic coordinates of the order country |
| Country longitude | Geographic coordinates of the order country |
| Region | Region to which the order belongs to |
| Market | Market to which the order belongs to |
| Subcategory | Product subcategory |
| Category | Product category |
| Product | Product name |
| Quantity | Number of products purchased per order |
| Sales | Total sales in $ |
| Discount | Discount applied to the order |
| Profit | Total profit for each order after discount |

{% enddetails %}


It consists of 19 columns, some of which are: 
- Identity features (Row ID, Order ID, Customer ID),
- Geographical features (Market, Region, Country, etc.), and 
- Product categorisation-related features (Category, Subcategory, etc.).

---

# Approach

To track the achievement of targets set out in the business problem, visualisations are used. Visualisations simplify complex data and allow us to identify patterns that help in informed decision-making. They also help track targets by making data more clear, interpretable, and actionable.

For creating visualisations and dashboard, I have used Tableau. Tableau is one of the most popular business intelligence tools available in the market. With its drag-and-drop interface, it presents a simple interface but also allows for more complex data analysis, real time updates and interactivity in dashboards.

---

# Insights

## General
1. The most profitable segment from 2020 to 2023 was `Corporate`, followed by `Consumer` segment.
2. The most profitable market between 2020 and 2023 was `Europe`. `Africa` remained the market with the least profits. Second to fourth position traded between the rest, with `Asia-Pacific` occupying the second position in 2020 and 2021, `Latin America` in 2022, and `United States and Canada` in 2023.
3. The best-selling product in 2020 was `Rose Gold Petal Studs`, followed by `Herbal Essences Bio`. In 2021, `Silver Frost Bracelet` occupied the top position, followed by `Herbal Essences Bio`. In 2022 and 2023, `Herbal Essences Bio` remained in the top spot.
4. The best-selling category between 2020 and 2023 is `Body Care` and the best-selling subcategory is `Nail care products`.

## Targets
### Annual Sales of at least $400K in each market

From 2020 to 2022, the annual sales target was surpassed only by `Asia-Pacific` once in 2022. In 2023, all markets except `Africa` have achieved the target. `Africa` achieved sales totalling $163k.

Average discount in `Africa` has hovered between 30% and 50%, which is much higher than the world average. With profit margin among the lowest in markets (13.6% as compared to the world’s 16.35%), the discount strategy has already been exhausted.

The average order value in `Africa` is $213, which is lower than the world average of $253. The proportion of one-time customers in `Africa` is 89.83%, which is much higher than the world average at 68.79%. Repeat customers tend to have a higher order value.

#### Recommendation 

Focus on increasing average order value by building customer loyalty. With customer’s making repeat purchases, average order value will increase.

### Annual Overall Sales Growth of 20% with a higher target of 30% for `Corporate` segment

The annual overall sales growth target of 20% was met in all years starting from 2021 till 2023. However, the `Corporate` sales growth target of 30% has not been achieved yet in any year. In the most recent 2 years (2022 and 2023), `Corporate` sales growth has hovered around 25%.

`Corporate` segment has a high base, with its sales surpassing `Consumer` and `Self-Employed` segments every year between 2020 and 2023. Within the `Corporate` segment, `Make up` category has shown growth potential, with its sales surpassing `Hair care` category in 2023. `Body care` remains the best-selling category while `Face care` is the least-selling category.

In subcategories, `Nail care products` (category: `Body care`) have been best-selling, while the second and third spots have been occupied by `Eye shadows and pencils` (category: `Make up`) and `Shampoos and conditioners` (category: `Hair care`).

The average order value within the `Corporate` segment is $497, much higher than the world average of $253.

The discount has varied between 20-25% which is on par with the average discounting strategy.

**Recommendation**

In order to increase sales, a higher discounting strategy for corporate segment can be explored.

### Average Annual Profit Margin of 15% across all Categories and Subcategories

`Home and Accessories` category has had a negative profit margin for all years between 2020 and 2023. `Hair care` category’s profit margin has been below 1% in the same period. These two remain the worst performing in terms of profit margin. In the remaining three categories, the average annual profit margin of 15% has been met in all years between 2020 and 2023. However, the profit margin for these three categories has been declining over the years.

`Fragrances` (category: `Home and Accessories`) have been the worst-performing subcategory with a profit margin remaining below (-)15% in all years between 2020 and 2023. Similarly, all other subcategories in `Home and Accessories`, and `Hair care` category have either had negative profit margin or a sub-1% profit margin in all these years. In body care, `face mask and exfoliators` and `body moisturisers` have had a negative profit margin. In all other subcategories profit margin has been above 15%.

**Recommendation**

In order to increase the overall profit margin, sales of `Home and Accessories` and `Hair care` category can be curtailed, or prices in these categories can be increased to boost the profit margin.

--- 

# Dashboard
<!-- <tableau-viz id="tableauViz"       
  src='https://public.tableau.com/shared/5MS3X77F9?:display_count=n&:origin=viz_share_link'>
</tableau-viz> -->

<div class='tableauPlaceholder' id='viz1741340600134' style='position: relative'><noscript><a href='#'><img alt='Product ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Be&#47;BeautyIncDashboard&#47;Product&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='BeautyIncDashboard&#47;Product' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Be&#47;BeautyIncDashboard&#47;Product&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-GB' /></object></div>                

<script type='text/javascript'>                    
var divElement = document.getElementById('viz1741340600134');                    
var vizElement = divElement.getElementsByTagName('object')[0];                    
if ( divElement.offsetWidth > 800 ) { vizElement.style.width='1000px';vizElement.style.height='827px';} 
else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='1000px';vizElement.style.height='827px';} 
else { vizElement.style.width='100%';vizElement.style.height='3077px';}                     
var scriptElement = document.createElement('script');                    
scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    
vizElement.parentNode.insertBefore(scriptElement, vizElement);                
</script>