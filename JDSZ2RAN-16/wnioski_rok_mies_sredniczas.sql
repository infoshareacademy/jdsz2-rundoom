select count(1)as ilosc_wnioskow, extract(year from w.data_utworzenia)as lata,date_part('month', w.data_utworzenia)as miesiac,
count(case when w.stan_wniosku = 'odrzucony prawnie' then w.id
 when w.stan_wniosku = 'odrzucony po analizie' then w.id
 when w.stan_wniosku = 'zamkniety' and w.kwota_rekompensaty = 0 then w.id
 when w.stan_wniosku = 'przegrany w sadzie' then w.id
 when w.stan_wniosku = 'odrzucony przez operatora' then w.id
 end)as odrzucone,
count(case when w.stan_wniosku = 'zaakceptowany przez operatora' then w.id
 when w.stan_wniosku = 'akcja sadowa' then w.id
  when w.stan_wniosku = 'nowy' then w.id
  when w.stan_wniosku = 'analiza zaakceptowana' then w.id
  when w.stan_wniosku = 'wysłany do operatora' then w.id
  end) as przetwarzany,
  count(case when w.stan_wniosku = 'wyplacony' then w.id
 when w.stan_wniosku = 'zamkniety' and w.kwota_rekompensaty >0 then w.id
  when w.stan_wniosku = 'wygrany w sadzie' then w.id
  end) as wyplacony,
  extract(epoch from avg(greatest(aw.data_zakonczenia, w.data_utworzenia)-least(aw.data_zakonczenia, w.data_utworzenia)))/3600 as sredni_czas_obslugi_wniosku
  from wnioski w
  inner join analizy_wnioskow aw on w.id = aw.id_wniosku
where extract(year from w.data_utworzenia)>'2013' and extract(year from w.data_utworzenia)<'2018'
group by extract(year from w.data_utworzenia),date_part('month', w.data_utworzenia)
order by lata
