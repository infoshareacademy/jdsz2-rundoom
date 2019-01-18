WITH DS as (
Select concat(date_part('year',data_utworzenia),'-',date_part('month',data_utworzenia)) as yyyy_mm,count(id) as ilosc_wnioskow_zlozonych,
sum(kwota_rekompensaty) as suma_rek_zlozonych,
round(avg(kwota_rekompensaty),2) as srednia_zlozonych
from wnioski
where (date_part('year',data_utworzenia) between 2014 and 2017) and stan_wniosku not like 'nowy'
group by date_part('month',data_utworzenia), date_part('year',data_utworzenia)
order by date_part('year',data_utworzenia),date_part('month',data_utworzenia)),


w_rek as (
Select concat(date_part('year',w.data_utworzenia),'-',date_part('month',w.data_utworzenia)),count(r.id_wniosku) as ilosc_wyplaconych,
sum(kwota_rekompensaty) as suma_rek,
round(avg(kwota_rekompensaty),2) as srednia_rekompensata
from rekompensaty r
left join wnioski w on r.id_wniosku = w.id
where date_part('year',w.data_utworzenia) between 2014 and 2017
group by 1)

Select yyyy_mm,ilosc_wnioskow_zlozonych,round(suma_rek_zlozonych/ilosc_wnioskow_zlozonych::numeric,2) as sr_kw_per_wn,
ilosc_wyplaconych,round(suma_rek/ilosc_wnioskow_zlozonych::numeric,2) as srednia_kw_wypl,
round(ilosc_wyplaconych/ilosc_wnioskow_zlozonych::numeric,2) as procent_wyplaconych
from DS
left join w_rek on DS.yyyy_mm = w_rek.concat


-- Select yyyy_mm,ilosc_wnioskow_zlozonych,suma_rek_zlozonych,ilosc_wyplaconych,suma_rek,abs(suma_rek_zlozonych - suma_rek) as roznica,
-- round(ilosc_wyplaconych/ilosc_wnioskow_zlozonych::numeric,2) as procent_wyplaconych
-- from DS
-- left join w_rek on DS.yyyy_mm = w_rek.concat


/*
Select *,srednia_zlozonych - srednia_rekompensata as roznica_rekompensat, round(ilosc_wyplaconych/ilosc_wnioskow_zlozonych::numeric,2) as procent_wyplaconych
from DS
left join w_rek on DS.concat = w_rek.concat*/



Select * from wnioski
where date_part('year', data_utworzenia) = 2016 and date_part('month',data_utworzenia) = 3



Select * from analizy_wnioskow a
left join rekompensaty r on r.id_wniosku = a.id_wniosku

Select r.id_wniosku,stan_wniosku from rekompensaty r
left join analizy_wnioskow a on r.id_wniosku = a.id_wniosku
left join wnioski w on r.id_wniosku = w.id
where a.id_wniosku is null

