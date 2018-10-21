/*Zapytanie pokazuje srednie czasy rozpatrywania wnioskow w podziale na lata/kwartaly*/
select date_part('year', w.data_utworzenia) as rok,
       date_part('quarter', w.data_utworzenia) as kwartal,
       avg(greatest(aw.data_zakonczenia, w.data_utworzenia)-least(aw.data_zakonczenia, w.data_utworzenia)) as sredni_czas_obslugi_wniosku
from wnioski w inner join analizy_wnioskow aw on w.id = aw.id_wniosku
where 1=1
    and aw.status like 'zaa%'
group by 1,2
order by 1,2;
