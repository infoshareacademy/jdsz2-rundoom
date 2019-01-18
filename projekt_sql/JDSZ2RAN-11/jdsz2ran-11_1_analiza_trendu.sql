WITH main_DS as (
Select date_part('month',data_utworzenia) as date_month,date_part('year',data_utworzenia) as date_year,count(id) as ilosc_wnioskow,
sum(
  case
  when lower(stan_wniosku)like 'wyplac%' then kwota_rekompensaty
  end
) as sum_wyplaconych_wnioskow
from wnioski
where date_part('year',data_utworzenia) between 2014 and 2017
group by data_utworzenia
order by data_utworzenia)
Select date_month,date_year,concat(date_month,'-',date_year),sum(ilosc_wnioskow) as ilosc_wnioskow,sum(sum_wyplaconych_wnioskow) as sum_wyyplaconych_rekompensat
from main_DS
group by date_year, date_month
order by date_year, date_month

-- zrobic zamiast like 'wyplac% ' dane z rekompensat i wywalic nowe
